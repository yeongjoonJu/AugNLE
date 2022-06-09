import json, os, random
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer, AutoFeatureExtractor

import utils

import logging
logger = logging.getLogger(__name__)

class BaseDataModule(LightningDataModule):
    def __init__(self, hparams, **kwargs,):
        super().__init__()
        sefl.save_hyperparameters(hparams)
        self.cfg = hparams

        random.seed(self.cfg.seed)

        self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.lm_backbone)
        # n_add_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<scene>', '<answer>']})

        # The number of few-shot data
        if hparams.fewshot_ratio > 0:
            self.fewshot_num = int(len(anno) * hparams.fewshot_ratio)
        else:
            self.fewshot_num = hparams.fewshot_num

        self.dataset = {}

        # Load dataset
        train_anno = json.load(open(hparams.train_anno_path, "r"))
        random.shuffle(train_anno)
        train_anno = train_anno[:self.fewshot_num]
        self.dataset["train"] = self.get_dataset(train_anno, is_train=True)

        valid_anno = json.load(open(hparams.valid_anno_path, "r"))
        self.dataset["valid"] = self.get_dataset(valid_anno, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.cfg.train_batch_size, pin_memory=True, num_workers=self.cfg.n_train_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], batch_size=self.cfg.train_batch_size, pin_memory=True, num_workers=self.cfg.n_valid_workers)

    def get_dataset(self, anno, is_train=False):
        raise NotImplementedError


class VQAXDataModule(BaseDataModule):
    def get_dataset(self, anno, is_train = False):
        cached_filename = f"vqax_shot-{self.fewshot_num}_seed-{self.cfg.seed}.cache"
        
        # Caching
        if os.path.exists(os.path.join(self.cfg.cached_dir, cached_filename)):
            logger.info("Loading features from cached file %s", cached_filename)
            features_and_dataset = torch.load(cached_features_file)
            datasets = features_and_dataset["datasets"]
        else:
            ids_list = list(anno.keys())
            index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
            for k,v in tqdm(anno.items(), desc= "Processing VQA-X annotation"):   
                if len(v['explanation']) > 1:   # some questions have more than one explanation 
                    ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

            # Set image directory
            if is_train:
                img_dir = self.img_data_dir + "/train2014/"
            else:
                img_dir = self.img_data_dir + "/val2014/"
                    
            datasets = []
            for i in tqdm(range(len(anno)), desc= "Processing VQA-X data"):
                question_id = ids_list[i]
                sample = data[question_id]
                img_name = sample['image_name']

                question_txt = utils.proc_ques(sample['question'])    # question
                answer_txt = utils.proc_ans(sample['answers'])
                exp_idx = index_tracker[question_id]
                explain_txt = sample['explanation'][exp_idx]

                # if one more explanations
                if exp_idx > 0:
                    index_tracker[question_id] -= 1    # decrease usage
                    
                # Question                
                question_input = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(question_txt))

                # Answer
                answer_input =  self.tokenizer.tokenize("answer: " + answer_txt)
                answer_input = tokenzier.convert_tokens_to_ids(answer_input)
                
                # scene
                # scene_tag = self.tokenizer.tokenize("scene:")

                # explanation
                explain_input = self.tokenizer.tokenize("reason: " + explain_txt)
                explain_input = self.tokenzier.convert_tokens_to_ids(explain_input)

                ## Image
                img_path = img_dir + img_name
                img = self.img_transform(Image.open(img_path))

                # Tensorize and add data
                datasets.append({
                    "Q": torch.tensor(question_input, dtype=torch.long),
                    "E": torch.tensor(explain_input, dtype=torch.long),
                    "A": torch.tensor(answer_input, dtype=torch.long),
                    "I": img, 
                })
            
            if not os.path.exists(self.cfg.cached_dir):
                os.mkdir(self.cfg.cached_dir)
            
            # save cached file
            torch.save(datasets, os.path.join(self.cfg.cached_dir, cached_filename))

        return datasets