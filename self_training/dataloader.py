import json, os, random
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pytorch_lightning import LightningDataModule
from transformers import GPT2Tokenizer

from models import utils

import logging
logger = logging.getLogger(__name__)

def random_data_choice(anno, num, pseudo=False):
    fewshot_data = {}
    # For pseudo labeling
    pseudo_data = {}
    
    img_ids = list(anno.keys())
    random.shuffle(img_ids)
    f_img_ids = img_ids[:num]
    p_img_ids = img_ids[num:]
    
    for img_id in f_img_ids:
        fewshot_data[img_id] = anno[img_id]
    if pseudo:
        for img_id in p_img_ids:
            pseudo_data[img_id] = anno[img_id]
    
    return fewshot_data


class BaseDataModule(LightningDataModule):
    def __init__(self, hparams, mode, turn, data_aug = False, ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.cfg = hparams

        self.img_transform = transforms.Compose([transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.cfg.lm_backbone)
        num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        self.dataset = {}

        # Load dataset
        train_anno = json.load(open(hparams.train_anno_path, "r"))
        
        # The number of few-shot data
        if hparams.fewshot_ratio > 0:
            self.fewshot_num = int(len(train_anno) * hparams.fewshot_ratio)
        else:
            self.fewshot_num = hparams.fewshot_num

        # Train dataset preprocessing
        train_anno = random_data_choice(train_anno, self.fewshot_num, pseudo=self.cfg.pseudo_data)
        dataset_name = "VQA-X"
        self.dataset["train"] = self.get_dataset(train_anno, mode=f"{dataset_name}_{mode}_train")
        
        # Data augmentation
        if data_aug:
            if os.path.exists(self.cfg.pseudo_labels_pth):
                dataset_name = "aug-data"
                aug_anno = json.load(open(self.cfg.pseudo_labels_pth, "r"))
                # VQA-X augmentation
                # self.dataset["train"] = self.dataset["train"] + self.get_dataset(aug_anno, mode=f"{dataset_name}_{mode}_train", data_aug = True)  
                # Caption data

                self.dataset["train"] = self.dataset["train"] + self.get_dataset(aug_anno, mode=f"{dataset_name}_{mode}_train", data_aug = True)
                

        valid_anno = json.load(open(hparams.valid_anno_path, "r"))
        dataset_name = "VQA-X"
        self.dataset["valid"] = self.get_dataset(valid_anno, mode=f"{dataset_name}_{mode}_valid")
        
        # captioning data
        caption_ann = json.load(open(hparams.captioning_pth, "r"))
        # self.dataset["pseudo"] = self.get_dataset(pseudo_ann, mode=mode+"_pseudo")
        
        # For test
        dataset_name = "nocaps"
        pseudo_data = self.get_dataset(caption_ann, mode=f"{dataset_name}_{mode}_pseudo")
        self.dataset["pseudo"] = utils.split_dataset(pseudo_data, len(pseudo_data)// hparams.iteration)[turn]
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.cfg.train_batch_size, \
                        pin_memory=True, num_workers=self.cfg.n_train_workers)#, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset["valid"], shuffle=False, batch_size=self.cfg.train_batch_size, \
                        pin_memory=True, num_workers=self.cfg.n_valid_workers)#, collate_fn=self.collate_fn)
        
    def predict_dataloader(self):
            return DataLoader(self.dataset["pseudo"], shuffle=False, batch_size= 1, \
                            pin_memory=True, num_workers=self.cfg.n_valid_workers)
            
    def test_dataloader(self):
            return DataLoader(self.dataset["pseudo"], shuffle=False, batch_size= 1, \
                            pin_memory=True, num_workers=self.cfg.n_valid_workers)

    
class VQAX_ST_DataModule(BaseDataModule):
    # Teacher >> [I] [Q] the answer is [A] --> <bos> because [E] <eos>
    # Student >> [I] [Q] --> <bos> the answer is [A] because [E] <eos>
    
    # Pseudo Labeling Teacher >> [I] [Q] the answer is [A] <bos> because --> Explanation
    
    # Pseudo Labeling Student >> [I] [Q] <bos> the answer is --> [A] because [E] <eos>


    def preprocessing(self, img_path, question, answer, explanation, mode):
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        # Image    
        # img = self.img_transform(Image.open(img_path).convert("RGB"), return_tensors="pt").pixel_values
        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)     
        if mode in ["train_student", "valid_student"]:
            input = self.tokenizer.tokenize(f"{question}")
            answer_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" the answer is {answer}")
            explanation_token = self.tokenizer.tokenize(f" because {explanation}") + [self.tokenizer.eos_token]
            segment_ids = [q_segment_id] * len(input) + [a_segment_id] * len(answer) + [e_segment_id] * len(explanation_token)
            output = answer_token + explanation_token   
            labels = [-100] * len(input) + [-100] + output[1:] # labels will be shifted in the model, so for now set them same as tokens
            input += output
            
            if len(input) > self.cfg.dec_max_len :
                input = input[:self.cfg.dec_max_len]
            labels = labels[:self.cfg.dec_max_len]
            segment_ids = segment_ids[:self.cfg.dec_max_len]
            
            # padddings
            input = input + ([self.tokenizer.pad_token] * (self.cfg.dec_max_len - len(input)))
            labels = labels + ([-100] * (self.cfg.dec_max_len - len(labels)))
            segment_ids += ([e_segment_id] * (self.cfg.dec_max_len - len(segment_ids)))       
            
        elif mode in ["train_teacher", "valid_teacher"]:
            question_token = self.tokenizer.tokenize(f"{question}")
            answer_token = self.tokenizer.tokenize(f" the answer is {answer}")
            output = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" because {explanation}") + [self.tokenizer.eos_token]
            segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len(answer_token) + [e_segment_id] * len(output)
            input = question_token + answer_token
            labels = [-100] * len(input) + [-100] + output[1:] # labels will be shifted in the model, so for now set them same as tokens
            input += output
            if len(input) > self.cfg.dec_max_len :
                input = input[:self.cfg.dec_max_len]
            labels = labels[:self.cfg.dec_max_len]
            segment_ids = segment_ids[:self.cfg.dec_max_len]
            
            # padddings
            input = input + ([self.tokenizer.pad_token] * (self.cfg.dec_max_len - len(input)))
            labels = labels + ([-100] * (self.cfg.dec_max_len - len(labels)))
            segment_ids += ([e_segment_id] * (self.cfg.dec_max_len - len(segment_ids)))   
            
        elif mode == "pseudo_student":
            input = self.tokenizer.tokenize(f"{question}")
            answer_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" the answer is")
            segment_ids = [q_segment_id] * len(input) + [a_segment_id] * len(answer_token)
            explanation_token = self.tokenizer.tokenize(explanation) + [self.tokenizer.eos_token]
            input += answer_token
            output = self.tokenizer.tokenize(f"{answer} {explanation}") + [self.tokenizer.eos_token]
            labels = [-100] * len(input) + output # labels will be shifted in the model, so for now set them same as tokens
        
        elif mode == "pseudo_teacher":
            question_token = self.tokenizer.tokenize(f"{question}")
            answer_token = self.tokenizer.tokenize(f" {answer}")
            explanation_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" because")
            input = question_token + answer_token + explanation_token
            output = self.tokenizer.tokenize(explanation) + [self.tokenizer.eos_token]
            segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len(answer_token) + [e_segment_id] * len(explanation_token)       
            labels = [-100] * len(input) + output # labels will be shifted in the model, so for now set them same as tokens
            
        else:
            raise NotImplementedError
        
        # token -> ids
        input_ids = self.tokenizer.convert_tokens_to_ids(input)

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        return input_ids, segment_ids, labels, img

    def get_dataset(self, anno, mode, data_aug=False):
        data_name, stage, mode = mode.split("_")
        cached_filename = f"{data_name}_shot-{self.fewshot_num}_{mode}_{stage}_seed-{self.cfg.seed}.cache"
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        
        # Caching
        if os.path.exists(os.path.join(self.cfg.cached_dir, cached_filename)) and not data_aug:
    
            logger.info("Loading features from cached file %s", cached_filename)
            datasets = torch.load(os.path.join(self.cfg.cached_dir, cached_filename))
        else:
            datasets = []
            
            # VQA-X dataset
            if data_name == "VQA-X":
                
                ids_list = list(anno.keys())
                index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
                for k,v in anno.items():   
                    if len(v['explanation']) > 1:   # some questions have more than one explanation 
                        ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

                # Set image directory
                if mode=="train":
                    img_dir = self.cfg.image_dir + "/train2014"
                    file_path = "train2014"
                else:
                    img_dir = self.cfg.image_dir + "/val2014"
                    file_path = "val2014"
                
                for i in tqdm(range(len(anno)), desc= f"Processing {data_name} {stage}-{mode} data"):
                    question_id = ids_list[i]
                    sample = anno[question_id]
                    img_name = sample['image_name']
                    image_id = sample["image_id"]
                    question_txt = utils.proc_ques(sample['question'])    # question
                    answer_txt = utils.proc_ans(sample['answers'])
                    exp_idx = index_tracker[question_id]
                    explain_txt = sample['explanation'][exp_idx]
                    img_pth = os.path.join(img_dir,img_name)
                    img_name = os.path.join(file_path,img_name)
                    
                    # if one more explanations
                    if exp_idx > 0:
                        index_tracker[question_id] -= 1    # decrease usage
                    
                    # Preprocessing dataset
                    input_ids, segment_ids, labels, img = \
                        self.preprocessing(img_pth, question_txt, answer_txt, explain_txt, f"{mode}_{stage}")
                    datasets.append({"qid": [img_name], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})  
                    
            # Captioning dataset
            elif data_name == "nocaps":
                for sample in tqdm(anno, desc= f"Processing {data_name} {stage}-{mode} data"):
                    img_name = os.path.join(data_name,sample['img_name'])
                    question_txt = utils.proc_ques(sample['question'])    # question
                    answer_txt = sample['answer']
                    explain_txt = sample['reason']
                    img_pth = os.path.join(self.cfg.image_dir, img_name)
                    input_ids, segment_ids, labels, img = \
                        self.preprocessing(img_pth, question_txt, answer_txt, explain_txt, f"{mode}_{stage}")
                    
                    qid = [img_name]
                    datasets.append({"qid": qid, "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})                    

            # Pseudo labeled data for augmentation
            elif data_name == "aug-data":
                
                ids_list = list(anno.keys())
                index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
                for k,v in anno.items():   
                    if len(v['explanation']) > 1:   # some questions have more than one explanation 
                        ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

                for i in tqdm(range(len(anno)), desc= f"Processing {data_name} {stage}-{mode} data"):
                    question_id = ids_list[i]
                    sample = anno[question_id]
                    img_name = sample["image_id"]
                    question_txt = utils.proc_ques(sample['question'])    # question
                    answer_txt = utils.proc_ans(sample['answers'])
                    exp_idx = index_tracker[question_id]
                    explain_txt = sample['explanation'][exp_idx]
                    img_pth = os.path.join(self.cfg.image_dir,img_name)

                    # if one more explanations
                    if exp_idx > 0:
                        index_tracker[question_id] -= 1    # decrease usage
                    
                    # Preprocessing dataset
                    input_ids, segment_ids, labels, img = \
                        self.preprocessing( img_pth, question_txt, answer_txt, explain_txt, f"{mode}_{stage}")
                    
                    datasets.append({"qid": [img_name], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})                    

            else:
                pass
            
            if not os.path.exists(self.cfg.cached_dir):
                os.mkdir(self.cfg.cached_dir)
            
            # save cached file
            if not data_aug:
                torch.save(datasets, os.path.join(self.cfg.cached_dir, cached_filename))
                
            # Not caching when data augmentation
            else:
                pass

        return datasets
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def get_dataset(self, anno, mode, data_aug=False):
#         data_name, stage, mode = mode.split("_")
#         cached_filename = f"{data_name}_shot-{self.fewshot_num}_{mode}_seed-{self.cfg.seed}.cache"
#         q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        
#         # Caching
#         if os.path.exists(os.path.join(self.cfg.cached_dir, cached_filename)) and not data_aug:
    
#             logger.info("Loading features from cached file %s", cached_filename)
#             datasets = torch.load(os.path.join(self.cfg.cached_dir, cached_filename))
#         else:
#             if data_name == "VQA-X":
                
#                 ids_list = list(anno.keys())
#                 index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
#                 for k,v in anno.items():   
#                     if len(v['explanation']) > 1:   # some questions have more than one explanation 
#                         ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

#                 # Set image directory
#                 if mode == "train" or mode == "valid":
#                     if mode=="train":
#                         img_dir = self.cfg.image_dir + "/train2014/"
#                     else:
#                         img_dir = self.cfg.image_dir + "/val2014/"
                            
#                     datasets = []
                    
#                     for i in tqdm(range(len(anno)), desc= f"Processing {data_name} {stage}-{mode} data"):
#                         question_id = ids_list[i]
#                         sample = anno[question_id]
#                         img_name = sample['image_name']
#                         image_id = sample["image_id"]
#                         question_txt = utils.proc_ques(sample['question'])    # question
#                         answer_txt = utils.proc_ans(sample['answers'])
#                         exp_idx = index_tracker[question_id]
#                         explain_txt = sample['explanation'][exp_idx]

#                         # if one more explanations
#                         if exp_idx > 0:
#                             index_tracker[question_id] -= 1    # decrease usage
#                         # Image    
#                         img_path = img_dir + img_name
#                         # img = self.img_transform(Image.open(img_path).convert("RGB"), return_tensors="pt").pixel_values
#                         img = Image.open(img_path).convert('RGB')
#                         img = self.img_transform(img)
                        
#                         question_token = self.tokenizer.tokenize(f"{question_txt}")
#                         answer_token = self.tokenizer.tokenize(f" the answer is {answer_txt}")
#                         explanation_token = self.tokenizer.tokenize(" because " + explain_txt)

#                         if stage == "student":
#                             input = question_token
#                             answer = [self.tokenizer.bos_token] + answer_token
#                             explanation = explanation_token + [self.tokenizer.eos_token]
#                             segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len(answer) + [e_segment_id] * len(explanation)
#                             output = answer + explanation                    
                            
#                         elif stage == "teacher":
#                             input = question_token + answer_token
#                             output = [self.tokenizer.bos_token] + explanation_token + [self.tokenizer.eos_token]
#                             segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len(answer_token) + [e_segment_id] * len(output)
                            
#                         else:
#                             raise NotImplementedError
                        
#                         labels = [-100] * len(input) + [-100] + output[1:] # labels will be shifted in the model, so for now set them same as tokens
#                         input += output
#                         if len(input) > self.cfg.dec_max_len :
#                                 input = input[:self.cfg.dec_max_len]
#                         labels = labels[:self.cfg.dec_max_len]
#                         segment_ids = segment_ids[:self.cfg.dec_max_len]
                        
#                         # padddings
#                         input = input + ([self.tokenizer.pad_token] * (self.cfg.dec_max_len - len(input)))
#                         labels = labels + ([-100] * (self.cfg.dec_max_len - len(labels)))
#                         segment_ids += ([e_segment_id] * (self.cfg.dec_max_len - len(segment_ids)))
                        
#                         # token -> ids
#                         input_ids = self.tokenizer.convert_tokens_to_ids(input)

#                         input_ids = torch.tensor(input_ids, dtype=torch.long)

#                         labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
#                         labels = torch.tensor(labels, dtype=torch.long)

#                         segment_ids = torch.tensor(segment_ids, dtype=torch.long)
#                         qid = torch.LongTensor([int(image_id)])
#                         datasets.append({"qid": qid, "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})
#                 else:
#                     img_dir = self.cfg.image_dir + "/val2014/"        
#                     datasets = []
#                     for i in tqdm(range(len(anno)), desc= f"Processing VQA-X {stage}-{mode} data"):
#                         question_id = ids_list[i]
#                         sample = anno[question_id]
#                         img_name = sample['image_name']

#                         question_txt = utils.proc_ques(sample['question'])    # question
#                         answer_txt = utils.proc_ans(sample['answers'])
#                         exp_idx = index_tracker[question_id]
#                         explain_txt = sample['explanation'][exp_idx]

#                         # if one more explanations
#                         if exp_idx > 0:
#                             index_tracker[question_id] -= 1    # decrease usage
#                         # Image    
#                         img_path = img_dir + img_name
#                         image_id = sample["image_id"]
#                         # img = self.img_transform(Image.open(img_path).convert("RGB"), return_tensors="pt").pixel_values
#                         img = Image.open(img_path).convert('RGB')
#                         img = self.img_transform(img)
                        
#                         question_token = self.tokenizer.tokenize(f"{question_txt}")
#                         answer_token = self.tokenizer.tokenize(f" the answer is")
#                         explanation_token = self.tokenizer.tokenize(" because")

#                         if stage == "student":
#                             input = question_token
#                             answer = [self.tokenizer.bos_token] + answer_token
#                             explanation = self.tokenizer.tokenize(f" because {explain_txt}") + [self.tokenizer.eos_token]
#                             segment_ids = [q_segment_id] * len(input) + [a_segment_id] * len(answer)
#                             input += answer
#                             output = self.tokenizer.tokenize(f" the answer is {answer_txt}") + explanation
                            
#                         elif stage == "teacher":
#                             answer = self.tokenizer.tokenize(f" the answer is {answer_txt}")
#                             input = question_token + answer + [self.tokenizer.bos_token] + explanation_token
#                             len_a = len(answer)
#                             len_e = len(explanation_token) + 1
#                             output = self.tokenizer.tokenize(explain_txt) + [self.tokenizer.eos_token]
#                             segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len_a + [e_segment_id] * len_e
                            
#                         else:
#                             raise NotImplementedError
                        
#                         labels = [-100] * len(input) + output # labels will be shifted in the model, so for now set them same as tokens
#                         if len(input) > self.cfg.dec_max_len :
#                                 input = input[:self.cfg.dec_max_len]
#                         labels = labels[:self.cfg.dec_max_len]
#                         segment_ids = segment_ids[:self.cfg.dec_max_len]
                        
                        
#                         # token -> ids
#                         input_ids = self.tokenizer.convert_tokens_to_ids(input)

#                         input_ids = torch.tensor(input_ids, dtype=torch.long)

#                         labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
#                         labels = torch.tensor(labels, dtype=torch.long)

#                         segment_ids = torch.tensor(segment_ids, dtype=torch.long)
#                         qid = torch.LongTensor([int(image_id)])
#                         datasets.append({"qid": qid, "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})                    
            
#             elif data_name == "nocaps":
#                 for sample in tqdm(anno, desc= f"Processing {data_name} {stage}-{mode} data"):
#                     img_name = sample['image_name']
#                     question_txt = utils.proc_ques(sample['question'])    # question
#                     answer_txt = utils.proc_ans(sample['answer'])
#                     explain_txt = sample['reason']

#             if not os.path.exists(self.cfg.cached_dir):
#                 os.mkdir(self.cfg.cached_dir)
            
#             # save cached file
#             if not data_aug:
#                 torch.save(datasets, os.path.join(self.cfg.cached_dir, cached_filename))
                
#             # Not caching when data augmentation
#             else:
#                 pass

#         return datasets