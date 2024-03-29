import json, os, random, re
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer, AutoFeatureExtractor

import utils

import logging
logger = logging.getLogger(__name__)

def random_data_choice(anno, num):
    fewshot_data = {}
    img_ids = list(anno.keys())
    random.shuffle(img_ids)
    img_ids = img_ids[:num]
    for img_id in img_ids:
        fewshot_data[img_id] = anno[img_id]
    
    return fewshot_data

class QuestionRefineDataset(Dataset):
    def __init__(self, pair_data, tokenizer):
        super().__init__()
        self.inputs = []
        self.labels = []
        self.img_names = []
        for i in tqdm(range(len(pair_data))):
            data = pair_data[i]
            question_txt = data["question"]    # question
            answer_txt = data["answer"]
            explain_txt = data["reason"]
            object_txt = data["objects"]

            # composition of text
            # [Q]? the answer is [A] because [E]
            t_n_input = f"{question_txt}? {answer_txt} {explain_txt}. {object_txt}"
            t_n_input = t_n_input.strip()

            # tokenize and encode
            t_n_input = tokenizer(t_n_input.lower()).input_ids
            self.inputs.append(t_n_input)
            self.img_names.append(data["img_name"])
    
    def __getitem__(self, index):
        enc_input = torch.tensor(self.inputs[index], dtype=torch.long)

        return enc_input, self.img_names[index]

    def __len__(self):
        return len(self.inputs)


class AnswerGenDataset(Dataset):
    def __init__(self, pair_data, tokenizer):
        super().__init__()
        self.inputs = []
        self.labels = []
        self.img_names = []
        self.objects = []
        for i in tqdm(range(len(pair_data))):
            data = pair_data[i]
            question_txt = data["question"]    # question
            answer_txt = data["answer"]
            explain_txt = data["reason"]

            # composition of text
            # question: [Q] reason: [E] -> the answer is [A]
            t_a_input = f"{question_txt}? reason: {explain_txt}"
            t_a_label = f"the answer is {answer_txt}"

            # tokenize and encode
            t_a_input = tokenizer(t_a_input.lower()).input_ids
            t_a_label = tokenizer(t_a_label.lower()).input_ids
            self.inputs.append(t_a_input)
            self.labels.append(t_a_label)
            self.img_names.append(data["img_name"])
            self.objects.append(data["objects"])
    
    def __getitem__(self, index):
        enc_input = torch.tensor(self.inputs[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return enc_input, label, self.img_names[index], self.objects[index]

    def __len__(self):
        return len(self.inputs)


class QAGenDataset(Dataset):
    def __init__(self, object_label_paths, tokenizer):
        super().__init__()

        data = {}
        for path in object_label_paths:
            with open(path, "r") as fin:
                data.update(json.load(fin))
            
        self.inputs = []
        self.img_names = []
        for img_name, label_pair in tqdm(data.items()):
            obj_label = label_pair["obj_label"]
            captions = label_pair["captions"]
            for caption in captions:
                # composition of text
                # because [E] -> [Q], the answer is [A]
                caption = re.sub(r"[.]", "", caption)
                t_e_input = f"because {caption}."
                t_e_input = f"{t_e_input} {obj_label}"
                t_e_input = t_e_input.strip()

                # tokenize and encode
                t_e_input = tokenizer(t_e_input.lower()).input_ids
                self.inputs.append(t_e_input)
                self.img_names.append(img_name)

    def __getitem__(self, index):
        enc_input = torch.tensor(self.inputs[index], dtype=torch.long)

        return enc_input, self.img_names[index]
    
    def __len__(self):
        return len(self.inputs)
        

class BaseDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.cfg = hparams

        # self.img_transform = transforms.Compose([transforms.Resize((hparams.img_size, hparams.img_size)),
        #                                          transforms.ToTensor(),
        #                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.img_transform = AutoFeatureExtractor.from_pretrained(self.cfg.visual_backbone)
        self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.lm_backbone)
        # n_add_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<scene>', '<answer>']})

        self.dataset = {}

        # Load dataset
        train_anno = json.load(open(hparams.train_anno_path, "r"))

        # The number of few-shot data
        if hparams.fewshot_ratio > 0:
            self.fewshot_num = int(len(train_anno) * hparams.fewshot_ratio)
        else:
            self.fewshot_num = hparams.fewshot_num

        train_anno = random_data_choice(train_anno, self.fewshot_num)
        self.dataset["train"] = self.get_dataset(train_anno, mode="train")

        valid_anno = json.load(open(hparams.valid_anno_path, "r"))
        self.dataset["valid"] = self.get_dataset(valid_anno, mode="val")

        # Collate function definition
        def collate_wrapper(batch):
            batch = list(zip(*batch))
            sample = {}

            # enc max len
            t_e_max_len = max([x.size(0) for x in batch[0]])
            t_a_max_len = max([x.size(0) for x in batch[2]])
            # t_n_max_len = max([x.size(0) for x in batch[4]])
            enc_max_len = max([t_e_max_len, t_a_max_len])

            # dec max len
            ex_max_len = max([x.size(0) for x in batch[1]])
            an_max_len = max([x.size(0) for x in batch[3]])
            # ac_max_len = max([x.size(0) for x in batch[5]])
            dec_max_len = max([ex_max_len, an_max_len])
            
            # t_e_input
            t_e_inputs = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
            t_e_attn_mask = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
            for i, x in enumerate(batch[0]):
                t_e_inputs[i,:x.size(0)] = x
                t_e_attn_mask[i,:x.size(0)] = 1.0
            
            # explanation (t_e_target)
            t_e_label = torch.zeros((len(batch[1]), dec_max_len), dtype=torch.long)
            for i, x in enumerate(batch[1]):
                t_e_label[i,:x.size(0)] = x

            # t_a_input
            t_a_inputs = torch.zeros((len(batch[2]), enc_max_len), dtype=torch.long)
            t_a_attn_mask = torch.zeros((len(batch[2]), enc_max_len), dtype=torch.long)
            for i, x in enumerate(batch[2]):
                t_a_inputs[i,:x.size(0)] = x
                t_a_attn_mask[i,:x.size(0)] = 1.0
            
            # answer (t_a_target)
            t_a_label = torch.zeros((len(batch[3]), dec_max_len), dtype=torch.long)
            for i, x in enumerate(batch[3]):
                t_a_label[i,:x.size(0)] = x

            # t_n_input
            # t_n_inputs = torch.zeros((len(batch[4]), enc_max_len), dtype=torch.long)
            # t_n_attn_mask = torch.zeros((len(batch[4]), enc_max_len), dtype=torch.long)
            # for i, x in enumerate(batch[4]):
            #     t_n_inputs[i,:x.size(0)] = x
            #     t_n_attn_mask[i,:x.size(0)] = 1.0
            
            # t_n_target
            # t_n_label = torch.zeros((len(batch[5]), dec_max_len), dtype=torch.long)
            # for i, x in enumerate(batch[5]):
            #     t_n_label[i,:x.size(0)] = x

            sample["enc_inputs"] = torch.cat((t_e_inputs, t_a_inputs), dim=0)
            sample["attn_mask"] = torch.cat((t_e_attn_mask, t_a_attn_mask), dim=0)
            sample["labels"] = torch.cat((t_e_label, t_a_label), dim=0)

            return sample

        self.collate_fn = collate_wrapper
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.cfg.train_batch_size, \
                        pin_memory=True, num_workers=self.cfg.n_train_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], shuffle=False, batch_size=self.cfg.train_batch_size, \
                        pin_memory=True, num_workers=self.cfg.n_valid_workers, collate_fn=self.collate_fn)
        
    def predict_dataloader(self):
            return DataLoader(self.dataset["pseudo"], shuffle=False, batch_size= 1, \
                            pin_memory=True, num_workers=self.cfg.n_valid_workers, collate_fn=self.collate_fn2)

    def get_dataset(self, anno, mode):
        raise NotImplementedError


class VQAXDataModule(BaseDataModule):
    # Main >> [I] question: [Q] -> the answer is [A] because [E]
    # T_e  >> because [E] -> question: [Q], the answer is [A]
    # T_a  >> question: [Q] reason: [E] -> the answer is [A]

    def get_dataset(self, anno, mode):
        cached_filename = f"vqax_shot-{self.fewshot_num}_{mode}_seed-{self.cfg.seed}.cache"
        
        # Caching
        if os.path.exists(os.path.join(self.cfg.cached_dir, cached_filename)):
            logger.info("Loading features from cached file %s", cached_filename)
            datasets = torch.load(os.path.join(self.cfg.cached_dir, cached_filename))
        else:
            ids_list = list(anno.keys())
            index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
            for k,v in anno.items():   
                if len(v['explanation']) > 1:   # some questions have more than one explanation 
                    ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

            obj_labels = None
            if self.cfg.object_label_dir is not None:
                obj_label_path = f"{self.cfg.object_label_dir}/obj_{mode}_labels.json"
                with open(obj_label_path, "r") as fin:
                    obj_labels = json.load(fin)
                    
            datasets = []
            question_ids = list(anno.keys())
            num_questions = len(question_ids)
            for i in tqdm(range(len(ids_list)), desc= f"Processing VQA-X {mode} data"):
                question_id = ids_list[i]
                sample = anno[question_id]
                img_name = sample['image_name']

                question_txt = utils.proc_ques(sample['question'])    # question
                answer_txt = utils.proc_ans(sample['answers'])
                exp_idx = index_tracker[question_id]
                explain_txt = sample['explanation'][exp_idx]

                # if one more explanations
                if exp_idx > 0:
                    index_tracker[question_id] -= 1    # decrease usage

                # composition of text
                # because [E] -> question: [Q], the answer is [A]
                t_e_input = f"because {explain_txt}. "
                t_e_label = f"{question_txt}? the answer is {answer_txt}"
                # question: [Q] reason: [E] -> the answer is [A]
                t_a_input = f"{question_txt}? reason: {explain_txt}"
                t_a_label = f"the answer is {answer_txt}"
                # [Q]|[Q'] the answer is [A]|[A'] because [E] -> acceptable | not acceptable quesion: [Q]
                # if random.random() < 0.5:
                #     t_n_input = f"{question_txt}? the answer is {answer_txt} {explain_txt}. "
                #     t_n_label = "acceptable"
                # else:
                #     n_q_id = question_ids[int(random.random()*num_questions)]
                #     neg_sample = anno[n_q_id]
                #     n_question_txt = utils.proc_ques(neg_sample['question'])    # question

                #     t_n_input = f"{n_question_txt}? the answer is {answer_txt} {explain_txt}. "
                #     t_n_label = f"not acceptable question: {question_txt}"

                if obj_labels is not None:
                    obj_label = obj_labels[img_name]
                    if obj_label:
                        t_e_input = t_e_input + obj_label
                        # t_n_input = t_n_input + obj_label
                    t_e_input = t_e_input.strip()
                    # t_n_input = t_n_input.strip()

                # tokenize and encode
                t_e_input = self.tokenizer(t_e_input).input_ids
                t_e_label = self.tokenizer(t_e_label).input_ids
                t_a_input = self.tokenizer(t_a_input).input_ids
                t_a_label = self.tokenizer(t_a_label).input_ids
                # t_n_input = self.tokenizer(t_n_input).input_ids
                # t_n_label = self.tokenizer(t_n_label).input_ids

                # Tensorize
                t_e_input = torch.tensor(t_e_input, dtype=torch.long)
                t_e_label = torch.tensor(t_e_label, dtype=torch.long)
                t_a_input = torch.tensor(t_a_input, dtype=torch.long)
                t_a_label = torch.tensor(t_a_label, dtype=torch.long)
                # t_n_input = torch.tensor(t_n_input, dtype=torch.long)
                # t_n_label = torch.tensor(t_n_label, dtype=torch.long)

                # add data
                datasets.append((t_e_input, t_e_label, t_a_input, t_a_label))
                
            
            if not os.path.exists(self.cfg.cached_dir):
                os.mkdir(self.cfg.cached_dir)
            
            # save cached file
            torch.save(datasets, os.path.join(self.cfg.cached_dir, cached_filename))

        return datasets
