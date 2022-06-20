import json, os, random
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pytorch_lightning import LightningDataModule
from transformers import T5Tokenizer, AutoFeatureExtractor

import utils

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
    
    return fewshot_data, pseudo_data


class BaseDataModule(LightningDataModule):
    def __init__(self, hparams, mode):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.cfg = hparams

        # self.img_transform = transforms.Compose([transforms.Resize((hparams.img_size, hparams.img_size)),
        #                                          transforms.ToTensor(),
        #                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.img_transform = AutoFeatureExtractor.from_pretrained(self.cfg.visual_backbone)
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

        train_anno, pseudo_ann = random_data_choice(train_anno, self.fewshot_num, pseudo=self.cfg.pseudo_data)
        self.dataset["train"] = self.get_dataset(train_anno, mode=mode+"_train")

        valid_anno = json.load(open(hparams.valid_anno_path, "r"))
        self.dataset["valid"] = self.get_dataset(valid_anno, mode=mode+"_valid")
        if self.cfg.pseudo_data:
            self.dataset["pseudo"] = self.get_dataset(pseudo_ann, mode=mode+"_pseudo")

        vis_rep_len = hparams.vis_rep_len
        
        if mode =="prompt":
        # Collate function definition
            def collate_wrapper(batch):
                batch = list(zip(*batch))
                sample = {}

                # enc max len
                t_e_max_len = max([x.size(0) for x in batch[0]])
                t_e_max_len += vis_rep_len
                t_a_max_len = max([x.size(0) for x in batch[2]])
                enc_max_len = t_e_max_len if t_e_max_len > t_a_max_len else t_a_max_len

                # dec max len
                ex_max_len = max([x.size(0) for x in batch[1]])
                an_max_len = max([x.size(0) for x in batch[3]])
                dec_max_len = ex_max_len if ex_max_len > an_max_len else an_max_len 
                
                # t_e_input
                t_e_inputs = torch.zeros((len(batch[0]), enc_max_len-vis_rep_len), dtype=torch.long)
                t_e_attn_mask = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
                for i, x in enumerate(batch[0]):
                    t_e_inputs[i,:x.size(0)] = x
                    t_e_attn_mask[i,:vis_rep_len+x.size(0)] = 1.0
                
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

                sample["enc_inputs"] = torch.cat((t_e_inputs, t_a_inputs), dim=0)
                sample["attn_mask"] = torch.cat((t_e_attn_mask, t_a_attn_mask), dim=0)
                sample["labels"] = torch.cat((t_e_label, t_a_label), dim=0)

                return sample
        else:
            prompt_len = len(self.tokenizer.tokenize(self.cfg.discrete_prompt))
            def collate_wrapper(batch):
                batch = list(zip(*batch))
                sample = {}
                vis_rep_len = self.cfg.vis_rep_len

                # enc max len
                enc_max_len = max([x.size(0) for x in batch[0]])
                # dec max len
                dec_max_len = max([x.size(0) for x in batch[1]])
                
                # enc_input
                enc_inputs = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
                enc_attn_mask = torch.zeros((len(batch[0]), prompt_len+vis_rep_len+enc_max_len), dtype=torch.long)
                for i, x in enumerate(batch[0]):
                    enc_inputs[i,:x.size(0)] = x
                    enc_attn_mask[i,:prompt_len+vis_rep_len+x.size(0)] = 1.0
                
                # label
                label = torch.zeros((len(batch[1]), dec_max_len), dtype=torch.long)
                for i, x in enumerate(batch[1]):
                    label[i,:x.size(0)] = x

                sample["enc_inputs"] = enc_inputs
                sample["enc_attn_mask"] = enc_attn_mask
                sample["label"] = label
                sample["img"] = torch.cat(batch[2])

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

    def get_dataset(self, anno, is_train=False):
        raise NotImplementedError


class VQAXDataModule(BaseDataModule):
    # Main >> [I] question: [Q] -> the answer is [A] because [E]
    # T_e  >> [I] question: [Q] answer: [A] -> because [E]
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

            # Set image directory
            stage, mode = mode.split("_")
            is_prompt = stage=="prompt"
            
            if mode=="train":
                img_dir = self.cfg.image_dir + "/train2014/"
            else:
                img_dir = self.cfg.image_dir + "/val2014/"
                    
            datasets = []
            for i in tqdm(range(len(anno)), desc= "Processing VQA-X {mode} data"):
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
                # Image    
                img_path = img_dir + img_name
                img = self.img_transform(Image.open(img_path).convert("RGB"), return_tensors="pt").pixel_values
                
                if is_prompt:
                    # composition of text
                    # [I] question: [Q] answer: [A] -> because [E]
                    t_e_input = f"because {explain_txt}"
                    t_e_label = f"question: {question_txt} , the answer is {answer_txt}"
                    # question: [Q] reason: [E] -> the answer is [A]
                    t_a_input = f"question: {question_txt} reason: {explain_txt}"
                    t_a_label = f"the answer is {answer_txt}"
                


                    # tokenize and encode
                    t_e_input = self.tokenizer(t_e_input).input_ids
                    t_e_label = self.tokenizer(t_e_label).input_ids
                    t_a_input = self.tokenizer(t_a_input).input_ids
                    t_a_label = self.tokenizer(t_a_label).input_ids

                    # Tensorize
                    t_e_input = torch.tensor(t_e_input, dtype=torch.long)
                    t_e_label = torch.tensor(t_e_label, dtype=torch.long)
                    t_a_input = torch.tensor(t_a_input, dtype=torch.long)
                    t_a_label = torch.tensor(t_a_label, dtype=torch.long)

                    # add data
                    datasets.append((t_e_input, t_e_label, t_a_input, t_a_label, img))
                    
                else:
                    enc_input = f"question: {question_txt}"
                    label = f"the answer is {answer_txt} because {explain_txt}"

                    # tokenize and encode
                    enc_input = self.tokenizer(enc_input).input_ids
                    label = self.tokenizer(label).input_ids

                    # Tensorize
                    enc_input = torch.tensor(enc_input, dtype=torch.long)
                    label = torch.tensor(label, dtype=torch.long)

                    # add data
                    datasets.append((enc_input, label, img))
                                        
            
            if not os.path.exists(self.cfg.cached_dir):
                os.mkdir(self.cfg.cached_dir)
            
            # save cached file
            torch.save(datasets, os.path.join(self.cfg.cached_dir, cached_filename))

        return datasets
