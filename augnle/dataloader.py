import torchvision.transforms as transforms
from transformers import T5Tokenizer
import json
import utils
import torch
from PIL import Image
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import os
import logging
from torch.utils.data import random_split
logger = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np

class VQAXDataModule(LightningDataModule):
    def __init__(
        self,
        hparams,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task_A = self.hparams.task_A
        self.img_size = self.hparams.img_size
        self.vqax_data_dir = self.hparams.annotation_data_dir
        self.coco_data_dir = self.hparams.coco_data_dir
        self.input_max_seq_length = self.hparams.input_max_seq_length#500
        self.output_max_seq_length = self.hparams.output_max_seq_length# 30
        self.train_batch_size = self.hparams.train_batch_size
        self.eval_batch_size = self.hparams.eval_batch_size        

        self.img_transform = transforms.Compose([transforms.Resize((self.img_size,self.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<scene>', '<answer>']})
        
        self.train_path = self.vqax_data_dir + "vqaX_train.json"
        self.val_path = self.vqax_data_dir + "vqaX_val.json"
        self.test_path = self.vqax_data_dir + "vqaX_test.json"
        
        
    def setup(self,stage=None):
        self.dataset = {}
        self.dataset["train"] = self.get_data(self.train_path, is_train = True)
        self.dataset["validation"] = self.get_data(self.val_path, is_train = False)
        #self.dataset["test"] = self.get_test_data(self.test_path)
        
        if self.hparams.fewshot is not None:
            np.random.seed(self.hparams.seed)
            few_length_train = int(len(self.dataset["train"]) * self.hparams.fewshot)
            few_length_val = int(len(self.dataset["validation"]) * self.hparams.fewshot)
            self.dataset["train"] = list(np.random.choice(self.dataset["train"], few_length_train))
            self.dataset["validation"] = list(np.random.choice(self.dataset["validation"], few_length_val))
        
    def get_data(self,data_path,is_train = False):
        cached_features_file = os.path.join(
            self.hparams.ckpt_path,# cache directory
            "cached_{}_{}".format(
                "train" if is_train else "dev",
                "VQAX",
            ),
        )
        if os.path.exists(cached_features_file):
                    logger.info("Loading features from cached file %s", cached_features_file)
                    features_and_dataset = torch.load(cached_features_file)
                    datasets = features_and_dataset["datasets"]
        else:
            data = json.load(open(data_path, 'r'))
            ids_list = list(data.keys())
            index_tracker = {k: len(v['explanation']) - 1 for k,v in data.items()}
            ids_list = list(data.keys())
            for k,v in tqdm(data.items(), desc= "Data to list and dictionary..."):   
                if len(v['explanation']) > 1:   # some questions have more than one explanation
            # duplicate them for loading. -1 because one explanation is already in ids_list
                    ids_list += [str(k)] * (len(v['explanation']) - 1)
                    
            datasets = []
            for i in tqdm(range(len(data)), desc= "VQA-X data preprocessing..."):
                quention_id = ids_list[i]
                sample = data[quention_id]
                img_name = sample['image_name']

                text_q = utils.proc_ques(sample['question'])    # question
                text_a = utils.proc_ans(sample['answers'])
                exp_idx = index_tracker[quention_id]
                text_e = sample['explanation'][exp_idx]

                # 2개의 explanation 이라면
                if exp_idx > 0:
                    index_tracker[quention_id] -= 1    # decrease usage
                    
                q_segment_id, s_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<scene>', '<answer>', '<explanation>'])

                question_tokens = self.tokenizer.tokenize(text_q)
                segment_ids = [q_segment_id] * len(question_tokens)

                # scene
                scene_tag = self.tokenizer.tokenize("scene:")

                # answer
                answer_tokens =  self.tokenizer.tokenize(text_a)
                answer_tag = self.tokenizer.tokenize("answer:")
                answer_len = len(answer_tokens)
                answer_tokens

                # explanation
                explanation_tokens = self.tokenizer.tokenize(text_e)
                explanation_tag = self.tokenizer.tokenize("explanation:")
                exp_len = len(explanation_tokens)

                # Task A -> Q, I, A -> E
                if self.task_A:
                    prompt_id = torch.tensor([1])
                    tokens = question_tokens  + answer_tag + answer_tokens  + scene_tag + ["<scene>"]*196 + [self.tokenizer.eos_token]
                    segment_ids = segment_ids + [s_segment_id]*196 + [e_segment_id] * (len(answer_tokens) +2)
                    labels = explanation_tokens + [self.tokenizer.eos_token]

                # Task B -> Q, E -> A
                else:
                    prompt_id = torch.tensor([0])
                    tokens = question_tokens + explanation_tag + explanation_tokens + [self.tokenizer.eos_token]
                    segment_ids = segment_ids + [e_segment_id] * (len(tokens) - len(segment_ids))
                    labels = answer_tokens + [self.tokenizer.eos_token]
                    
                # Split over sequence length
                if len(tokens) > self.input_max_seq_length :
                    tokens = tokens[:self.input_max_seq_length]
                    segment_ids = segment_ids[:self.input_max_seq_length]
                    
                # Padding
                attention_mask = torch.zeros(self.input_max_seq_length)
                seq_len = len(tokens)
                attention_mask[:seq_len] = torch.tensor([1])
                input_padding_len = self.input_max_seq_length - len(tokens)
                output_padding_len = self.output_max_seq_length - len(labels)
                tokens = tokens + ([self.tokenizer.pad_token] * input_padding_len)
                labels = labels + ([self.tokenizer.pad_token] * output_padding_len)
                segment_ids += ([e_segment_id] * input_padding_len)
                # token to ids

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor(input_ids, dtype=torch.long)

                labels = [self.tokenizer.convert_tokens_to_ids(t) for t in labels]
                labels = torch.tensor(labels, dtype=torch.long)

                segment_ids = torch.tensor(segment_ids, dtype=torch.long)


                ## Image

                folder = self.coco_data_dir + '/train2014/' if 'train' in img_name else self.coco_data_dir + 'val2014/'
                img_path = folder + img_name
                img = Image.open(img_path).convert('RGB')
                img = self.img_transform(img)
                qid =quention_id
                qid = torch.LongTensor([int(quention_id)])

                datasets.append({"prompt_id": prompt_id, "img": img, "qid" : qid, "input_ids": input_ids, "labels": labels, "segment_ids" : segment_ids, "attention_mask" : attention_mask})
            torch.save({"datasets": datasets}, cached_features_file)

        return datasets
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.train_batch_size, pin_memory=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], shuffle=True, batch_size=self.train_batch_size, pin_memory=True, num_workers=self.hparams.num_workers)
    
    #def test_dataloader(self):
    #    return DataLoader(self.dataset["test"], batch_size=self.train_batch_size, pin_memory=True, num_workers=self.hparams.num_workers)
