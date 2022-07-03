from asyncio import DatagramTransport
import json, os, random
from xml.etree.ElementInclude import include
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
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


class VQAXEvalDataset(Dataset):
    def __init__(self, path, img_dir, tokenizer, args):

        self.tokenizer = tokenizer
        self.transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.max_seq_len = 40       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.img_dir = img_dir

    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = utils.proc_ques(sample['question'])    # question

        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(answer)
        tokens += answer 

        segment_ids += [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])
        
        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


class ExGenDataset(Dataset):
    def __init__(self, relabeled_data):
        super().__init__()

        for sample in relabeled_data:
            pass
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


class BaseDataset(Dataset):
    def __init__(self, image_dir, nle_anno_path, tokenizer, cfg, include_captioning=False, teacher_mode=False):
        super().__init__()
        self.cfg = cfg

        self.teacher_mode = teacher_mode
        self.img_transform = transforms.Compose([transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.tokenizer = tokenizer

        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        self.q_seg_id = q_segment_id
        self.a_seg_id = a_segment_id
        self.e_seg_id = e_segment_id
        self.answer_split = self.tokenizer(" the answer is ").input_ids
        self.reason_split = self.tokenizer(" because ").input_ids
        self.question_split = self.tokenizer("?").input_ids[0]
        mode = "teacher" if teacher_mode else "student"

        # Load cache
        cached_path = os.path.join(self.cfg.cached_dir, f"total_{mode}.cache")
        if os.path.exists(cached_path):
            logger.info("Loading features from cached file %s", cached_path)
            self.datasets = torch.load(cached_path)
            with open(os.path.join(self.cfg.cached_dir, f"nle_ids_data.json"), "r") as fin:
                self.nle_ids_data = json.load(fin)
            if include_captioning:
                with open(os.path.join(self.cfg.cached_dir, f"relabeled_ids_data.json"), "r") as fin:
                    self.relabeled_ids_data = json.load(fin)
            else:
                self.relabeled_ids_data = []
        else:
            # Captioning data
            if include_captioning:
                assert len(cfg.captioning_image_dirs)==len(cfg.captioning_anno_paths)
                self.captioning_data = []
                for img_dir, anno_path in zip(cfg.captioning_image_dirs, cfg.captioning_anno_paths):
                    with open(anno_path, "r") as fin:
                        anno = json.load(fin)
                    for sample in anno:
                        sample["img_name"] = os.path.join(img_dir, sample["img_name"])
                        self.captioning_data.append(sample)
            
            with open(nle_anno_path, "r") as fin:
                anno = json.load(fin)
            self.datasets, self.nle_ids_data, self.relabeled_ids_data \
                = self.get_dataset(image_dir, anno, include_captioning=include_captioning)

            if not os.path.exists(self.cfg.cached_dir):
                os.mkdir(self.cfg.cached_dir)

            # Caching            
            torch.save(self.datasets, cached_path)
            with open(os.path.join(self.cfg.cached_dir, f"nle_ids_data.json"), "w") as fout:
                json.dump(self.nle_ids_data, fout, indent=2)
            if self.relabeled_ids_data:
                with open(os.path.join(self.cfg.cached_dir, f"relabeled_ids_data.json"), "w") as fout:
                    json.dump(self.relabeled_ids_data, fout, indent=2)
    
    def get_collate_fn(self):
        # Collate function definition
        def collate_wrapper(batch):
            batch = list(zip(*batch))
            sample = {}

            # max len
            max_len = max([x.size(0) for x in batch[0]])

            inputs = torch.zeros((len(batch[0]), max_len), dtype=torch.long) + self.tokenizer.pad_token_id
            # attn_mask = torch.zeros((len(batch[0]), max_len), dtype=torch.long)
            labels = torch.zeros((len(batch[1]), max_len), dtype=torch.long) - 100
            seg_ids = torch.zeros((len(batch[2]), max_len), dtype=torch.long) + self.e_seg_id
            for i, x in enumerate(batch[0]):
                inputs[i,:x.size(0)] = x
                # attn_mask[i,:x.size(0)] = 1.0
                labels[i,:x.size(0)] = batch[1][i]
                seg_ids[i,:x.size(0)] = batch[2][i]

            sample["inputs"] = inputs
            # sample["attn_mask"] = attn_mask
            sample["labels"] = labels
            sample["segment_ids"] = seg_ids
            sample["image"] = torch.stack(batch[3])
            sample["image_path"] = batch[4]

            return sample
        
        return collate_wrapper

    def __getitem__(self, index):
        sample = self.datasets[index]
        return sample["input_ids"], sample["labels"], sample["segment_ids"], sample["img"], sample["image_path"]
    
    def __len__(self):
        return len(self.datasets)

    def get_dataset(self, image_dir, anno, include_captioning=False):
        raise NotImplementedError

    def preprocess(self, img_path, question_ids, answer_ids, explanation_ids):
        raise NotImplementedError

    def get_relabeled_targets(self):
        raise self.relabeled_ids_data

    def change_mode(self, mode):
        if (self.teacher_mode and mode=="teacher") or (not self.teacher_mode and mode=="student"):
            return
        if mode=="teacher":
            self.teacher_mode = True
        else:
            self.teacher_mode = False

        cached_path = os.path.join(self.cfg.cached_dir, f"total_{mode}.cache")
        if os.path.exists(cached_path):
            logger.info("Loading features from cached file %s", cached_path)
            self.datasets = torch.load(cached_path)
        else:
            # Preprocessing dataset
            print(len(self.nle_ids_data), len(self.relabeled_ids_data))
            self.datasets = []
            for sample in tqdm(self.nle_ids_data+self.relabeled_ids_data, desc=f"Changing to {mode} mode"):
                input_ids, segment_ids, labels, img = self.preprocess(
                    sample["image_path"], sample["question"], sample["answer"], sample["explain"])
                self.datasets.append({"image_path": sample["image_path"], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})
            
            torch.save(self.datasets, cached_path)


class VQAX_full_shot_Dataset(BaseDataset):
    def get_dataset(self, image_dir, anno, include_captioning=False):
        ids_list = list(anno.keys())
        index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
        for k,v in anno.items():   
            if len(v['explanation']) > 1:   # some questions have more than one explanation 
                ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

        nle_data = []
        for i in tqdm(range(len(ids_list)), desc="Constructing NLE data"):
            question_id = ids_list[i]
            sample = anno[question_id]
            img_name = sample['image_name']
            question_txt = utils.proc_ques(sample['question'])    # question
            answer_txt = utils.proc_ans(sample['answers'])
            exp_idx = index_tracker[question_id]
            img_path = os.path.join(image_dir, img_name)

            # if one more explanations
            if exp_idx > 0:
                index_tracker[question_id] -= 1    # decrease usage

            explain_txt = sample['explanation'][exp_idx]

            question = self.tokenizer(question_txt).input_ids
            answer = self.tokenizer(answer_txt).input_ids
            explain = self.tokenizer(explain_txt).input_ids

            nle_data.append({"image_path": img_path, "question": question, "answer": answer, "explain": explain})

        relabeled_data = []
        if include_captioning:
            for i in tqdm(range(len(self.captioning_data)), desc="Constructing Captioning data"):
                sample = self.captioning_data[i]
                question = self.tokenizer(sample["question"]).input_ids
                answer = self.tokenizer(sample["answer"]).input_ids
                explain = self.tokenizer(sample["reason"]).input_ids
                relabeled_data.append({
                    "image_path": sample["img_name"], "question": question,
                    "answer": answer, "explain": explain})

        data = []    
        # Preprocessing dataset
        for sample in tqdm(nle_data+relabeled_data, desc="Tokenizing and Tensorizing"):
            input_ids, segment_ids, labels, img = self.preprocess(
                sample["image_path"], sample["question"], sample["answer"], sample["explain"])
            data.append({"image_path": sample["image_path"], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})

        return data, nle_data, relabeled_data
    
    def preprocess(self, img_path, question_ids, answer_ids, explanation_ids):
        # Image    
        # img = self.img_transform(Image.open(img_path).convert("RGB"), return_tensors="pt").pixel_values
        img = Image.open(img_path).convert('RGB')

        if self.teacher_mode:
            input_ids = question_ids.copy()
            input_ids.append(self.question_split)
            input_ids.extend(self.answer_split.copy())
            input_ids.extend(answer_ids)
            output = [self.tokenizer.bos_token_id]
            output.extend(self.reason_split.copy())
            output.extend(explanation_ids)
            output.append(self.tokenizer.eos_token_id)
            segment_ids = [self.q_seg_id]*(len(question_ids)+1)
            segment_ids.extend([self.a_seg_id]*(len(answer_ids)+len(self.answer_split)))
            segment_ids.extend([self.e_seg_id]*len(output))
        else:
            input_ids = question_ids.copy()
            output_a = [self.tokenizer.bos_token_id]
            output_a.extend(self.answer_split.copy())
            output_a.extend(answer_ids)
            output_e = self.reason_split.copy()
            output_e.extend(explanation_ids)
            output_e.append(self.tokenizer.eos_token_id)
            segment_ids = [self.q_seg_id]*len(input_ids)
            segment_ids.extend([self.a_seg_id]*len(output_a))
            segment_ids.extend([self.e_seg_id]*len(output_e))
            output = output_a + output_e

        labels = [-100]*len(input_ids)
        labels.append(-100)
        labels.extend(output[1:]) # labels will be shifted in the model, so for now set them same as tokens
        input_ids.extend(output)

        # tensorize
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        img = self.img_transform(img)
        
        return input_ids, segment_ids, labels, img



# class VQAX_few_shot_DataModule(BaseDataModule):
#     def get_dataset(self):
#         # Caching
#         if os.path.exists(os.path.join(self.cfg.cached_dir, cached_filename)) and not data_aug:
#             logger.info("Loading features from cached file %s", cached_filename)
#             datasets = torch.load(os.path.join(self.cfg.cached_dir, cached_filename))
#         else:
#             # The number of few-shot data
#             if self.cfg.fewshot_ratio > 0:
#                 self.fewshot_num = int(len(self.train_anno) * self.cfg.fewshot_ratio)
#             elif self.cfg.fewshot_num is None:
#                 # external use
#                 pass
                
#             # Train dataset preprocessing
#             train_anno = random_data_choice(train_anno, self.fewshot_num, pseudo=self.cfg.pseudo_data)
#             self.dataset["train"] = self.get_dataset(train_anno, mode="train")
    
            
    
# class VQAX_ST_DataModule(BaseDataModule):
#     # Teacher >> [I] [Q] the answer is [A] --> <bos> because [E] <eos>
#     # Student >> [I] [Q] --> <bos> the answer is [A] because [E] <eos>
    
#     # Pseudo Labeling Teacher >> [I] [Q] the answer is [A] <bos> because --> Explanation
    
#     # Pseudo Labeling Student >> [I] [Q] <bos> the answer is --> [A] because [E] <eos>


#     def preprocessing(self, img_path, question, answer, explanation, mode):
#         q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
#         # Image    
#         # img = self.img_transform(Image.open(img_path).convert("RGB"), return_tensors="pt").pixel_values
#         img = Image.open(img_path).convert('RGB')
#         img = self.img_transform(img)     
#         if mode in ["train_student", "valid_student"]:
#             input = self.tokenizer.tokenize(f"{question}")
#             answer_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" the answer is {answer}")
#             explanation_token = self.tokenizer.tokenize(f" because {explanation}") + [self.tokenizer.eos_token]
#             segment_ids = [q_segment_id] * len(input) + [a_segment_id] * len(answer) + [e_segment_id] * len(explanation_token)
#             output = answer_token + explanation_token   
#             labels = [-100] * len(input) + [-100] + output[1:] # labels will be shifted in the model, so for now set them same as tokens
#             input += output
            
#             if len(input) > self.cfg.dec_max_len :
#                 input = input[:self.cfg.dec_max_len]
#             labels = labels[:self.cfg.dec_max_len]
#             segment_ids = segment_ids[:self.cfg.dec_max_len]
            
#             # padddings
#             input = input + ([self.tokenizer.pad_token] * (self.cfg.dec_max_len - len(input)))
#             labels = labels + ([-100] * (self.cfg.dec_max_len - len(labels)))
#             segment_ids += ([e_segment_id] * (self.cfg.dec_max_len - len(segment_ids)))       
            
#         elif mode in ["train_teacher", "valid_teacher"]:
#             question_token = self.tokenizer.tokenize(f"{question}")
#             answer_token = self.tokenizer.tokenize(f" the answer is {answer}")
#             output = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" because {explanation}") + [self.tokenizer.eos_token]
#             segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len(answer_token) + [e_segment_id] * len(output)
#             input = question_token + answer_token
#             labels = [-100] * len(input) + [-100] + output[1:] # labels will be shifted in the model, so for now set them same as tokens
#             input += output
#             if len(input) > self.cfg.dec_max_len :
#                 input = input[:self.cfg.dec_max_len]
#             labels = labels[:self.cfg.dec_max_len]
#             segment_ids = segment_ids[:self.cfg.dec_max_len]
            
#             # padddings
#             input = input + ([self.tokenizer.pad_token] * (self.cfg.dec_max_len - len(input)))
#             labels = labels + ([-100] * (self.cfg.dec_max_len - len(labels)))
#             segment_ids += ([e_segment_id] * (self.cfg.dec_max_len - len(segment_ids)))   
            
#         elif mode == "pseudo_student":
#             input = self.tokenizer.tokenize(f"{question}")
#             answer_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" the answer is")
#             segment_ids = [q_segment_id] * len(input) + [a_segment_id] * len(answer_token)
#             explanation_token = self.tokenizer.tokenize(explanation) + [self.tokenizer.eos_token]
#             input += answer_token
#             output = self.tokenizer.tokenize(f"{answer} {explanation}") + [self.tokenizer.eos_token]
#             labels = [-100] * len(input) + output # labels will be shifted in the model, so for now set them same as tokens
        
#         elif mode == "pseudo_teacher":
#             question_token = self.tokenizer.tokenize(f"{question}")
#             answer_token = self.tokenizer.tokenize(f" {answer}")
#             explanation_token = [self.tokenizer.bos_token] + self.tokenizer.tokenize(f" because")
#             input = question_token + answer_token + explanation_token
#             output = self.tokenizer.tokenize(explanation) + [self.tokenizer.eos_token]
#             segment_ids = [q_segment_id] * len(question_token) + [a_segment_id] * len(answer_token) + [e_segment_id] * len(explanation_token)       
#             labels = [-100] * len(input) + output # labels will be shifted in the model, so for now set them same as tokens
            
#         else:
#             raise NotImplementedError
        
#         # token -> ids
#         input_ids = self.tokenizer.convert_tokens_to_ids(input)

#         input_ids = torch.tensor(input_ids, dtype=torch.long)

#         labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
#         labels = torch.tensor(labels, dtype=torch.long)
#         segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
#         return input_ids, segment_ids, labels, img

#     def get_dataset(self, anno, mode, data_aug=False):
#         data_name, stage, mode = mode.split("_")
#         cached_filename = f"{data_name}_shot-{self.fewshot_num}_{mode}_{stage}_seed-{self.cfg.seed}.cache"
#         q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        
#         # Caching
#         if os.path.exists(os.path.join(self.cfg.cached_dir, cached_filename)) and not data_aug:
    
#             logger.info("Loading features from cached file %s", cached_filename)
#             datasets = torch.load(os.path.join(self.cfg.cached_dir, cached_filename))
#         else:
#             datasets = []
            
#             # VQA-X dataset
#             if data_name == "VQA-X":
                
#                 ids_list = list(anno.keys())
#                 index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
#                 for k,v in anno.items():   
#                     if len(v['explanation']) > 1:   # some questions have more than one explanation 
#                         ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

#                 # Set image directory
#                 if mode=="train":
#                     img_dir = self.cfg.image_dir + "/train2014"
#                 else:
#                     img_dir = self.cfg.image_dir + "/val2014"
                
#                 for i in tqdm(range(len(anno)), desc= f"Processing {data_name} {stage}-{mode} data"):
#                     question_id = ids_list[i]
#                     sample = anno[question_id]
#                     img_name = sample['image_name']
#                     image_id = sample["image_id"]
#                     question_txt = utils.proc_ques(sample['question'])    # question
#                     answer_txt = utils.proc_ans(sample['answers'])
#                     exp_idx = index_tracker[question_id]
#                     explain_txt = sample['explanation'][exp_idx]
#                     img_pth = os.path.join(img_dir,img_name)

#                     # if one more explanations
#                     if exp_idx > 0:
#                         index_tracker[question_id] -= 1    # decrease usage
                    
#                     # Preprocessing dataset
#                     input_ids, segment_ids, labels, img = \
#                         self.preprocessing(img_pth, question_txt, answer_txt, explain_txt, f"{mode}_{stage}")
                    
#                     datasets.append({"qid": [image_id], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})    
                    
#             # Captioning dataset
#             elif data_name == "nocaps":
#                 for sample in tqdm(anno, desc= f"Processing {data_name} {stage}-{mode} data"):
#                     img_name = os.path.join(data_name,sample['img_name'])
#                     question_txt = utils.proc_ques(sample['question'])    # question
#                     answer_txt = sample['answer']
#                     explain_txt = sample['reason']
#                     img_pth = os.path.join(self.cfg.image_dir, img_name)
#                     input_ids, segment_ids, labels, img = \
#                         self.preprocessing(img_pth, question_txt, answer_txt, explain_txt, f"{mode}_{stage}")
                    
#                     qid = [img_name]
#                     datasets.append({"qid": qid, "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})                    

#             # Pseudo labeled data for augmentation
#             elif data_name == "aug-data":
                
#                 ids_list = list(anno.keys())
#                 index_tracker = {k: len(v['explanation']) - 1 for k,v in anno.items()}
#                 for k,v in anno.items():   
#                     if len(v['explanation']) > 1:   # some questions have more than one explanation 
#                         ids_list += [str(k)] * (len(v['explanation']) - 1) # duplicate them for loading. -1 because one explanation is already in ids_list

#                 for i in tqdm(range(len(anno)), desc= f"Processing {data_name} {stage}-{mode} data"):
#                     question_id = ids_list[i]
#                     sample = anno[question_id]
#                     img_name = sample["image_id"]
#                     question_txt = utils.proc_ques(sample['question'])    # question
#                     answer_txt = utils.proc_ans(sample['answers'])
#                     exp_idx = index_tracker[question_id]
#                     explain_txt = sample['explanation'][exp_idx]
#                     img_pth = os.path.join(self.cfg.image_dir,img_name)

#                     # if one more explanations
#                     if exp_idx > 0:
#                         index_tracker[question_id] -= 1    # decrease usage
                    
#                     # Preprocessing dataset
#                     input_ids, segment_ids, labels, img = \
#                         self.preprocessing( img_pth, question_txt, answer_txt, explain_txt, f"{mode}_{stage}")
                    
#                     datasets.append({"qid": [img_name], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids, "img":img})                    

#             else:
#                 pass
            
#             if not os.path.exists(self.cfg.cached_dir):
#                 os.mkdir(self.cfg.cached_dir)
            
#             # save cached file
#             if not data_aug:
#                 torch.save(datasets, os.path.join(self.cfg.cached_dir, cached_filename))
                
#             # Not caching when data augmentation
#             else:
#                 pass

#         return datasets
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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