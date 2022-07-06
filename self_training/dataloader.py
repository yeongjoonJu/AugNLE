import json, os, random
from tqdm import tqdm
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
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
        self.q_seg_id, self.a_seg_id, self.e_seg_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])

    def __getitem__(self, i):
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        text_a = utils.proc_ques(sample['question'])    # question

        # tokenization process
        input_ids = self.tokenizer(text_a).input_ids
        segment_ids = [self.q_seg_id] * len(input_ids)

        answer = [self.tokenizer.bos_token_id]
        answer.extend(self.tokenizer(" the answer is ").input_ids)
        answer_len = len(answer)
        input_ids.extend(answer)

        segment_ids.extend([self.a_seg_id] * answer_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])
        
        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


class VQAXPredDataset(Dataset):
    def __init__(self, data, tokenizer, args):

        self.transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.max_seq_len = 40       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = data
        self.img_encoded = args.img_encoded
        self.bos_token_id = tokenizer.bos_token_id
        self.q_seg_id, self.a_seg_id, self.e_seg_id = tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        self.answer_split = tokenizer(" the answer is ").input_ids

    def __getitem__(self, i):
        sample = {}
        input_ids = self.data[i]["question"]
        segment_ids = [self.q_seg_id] * len(input_ids)
        answer = [self.bos_token_id]
        answer.extend(self.answer_split.copy())
        answer_len = len(answer)
        input_ids.extend(answer)
        segment_ids.extend([self.a_seg_id] * answer_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        if self.img_encoded:
            img_path = self.data[i]["image_path"]
            # dir_name = os.path.dirname(img_path) + "_encoded"
            # filename = os.path.basename(img_path) + ".npy"
            # img_path = os.path.join(dir_name, filename)
            img = torch.from_numpy(np.load(img_path))
        else:
            img_path = self.data[i]["image_path"]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        
        return input_ids, segment_ids, img, self.data[i]["idx"]

    def __len__(self):
        return len(self.data)
    
    def get_collate_fn(self):
        # Collate function definition
        def collate_wrapper(batch):
            batch = list(zip(*batch))
            sample = {}

            # max len
            max_len = max([x.size(0) for x in batch[0]])

            inputs = torch.zeros((len(batch[0]), max_len), dtype=torch.long) - 1
            seg_ids = torch.zeros((len(batch[1]), max_len), dtype=torch.long) - 1
            for i, x in enumerate(batch[0]):
                inputs[i,:x.size(0)] = x
                seg_ids[i,:x.size(0)] = batch[1][i]
            
            sample["input_ids"] = inputs
            sample["segment_ids"] = seg_ids
            if self.img_encoded:
                sample["img_embeddings"] = torch.stack(batch[2])
            else:
                sample["image"] = torch.stack(batch[2])
            
            sample["id"] = batch[3]

            return sample
        
        return collate_wrapper
    

class BaseDataset(Dataset):
    def __init__(self, image_dir, nle_anno_path, tokenizer, cfg, include_captioning=False, teacher_mode=False):
        super().__init__()
        self.cfg = cfg

        self.teacher_mode = teacher_mode
        self.img_encoded = cfg.img_encoded
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
                        if self.img_encoded:
                            sample["img_name"] = os.path.join(img_dir+"_encoded", sample["img_name"]+".npy")
                        else:
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
            slicing = False
            max_len = max([x.size(0) for x in batch[0]])
            if self.cfg.max_seq_len < max_len:
                max_len = self.cfg.max_seq_len
                slicing = True

            inputs = torch.zeros((len(batch[0]), max_len), dtype=torch.long) + self.tokenizer.pad_token_id
            # attn_mask = torch.zeros((len(batch[0]), max_len), dtype=torch.long)
            labels = torch.zeros((len(batch[1]), max_len), dtype=torch.long) - 100
            seg_ids = torch.zeros((len(batch[2]), max_len), dtype=torch.long) + self.e_seg_id
            for i, x in enumerate(batch[0]):
                if slicing:
                    x = x[:max_len]
                inputs[i,:x.size(0)] = x
                # attn_mask[i,:x.size(0)] = 1.0
                labels[i,:x.size(0)] = batch[1][i][:max_len] if slicing else batch[1][i]
                seg_ids[i,:x.size(0)] = batch[2][i][:max_len] if slicing else batch[2][i]

            sample["inputs"] = inputs
            # sample["attn_mask"] = attn_mask
            sample["labels"] = labels
            sample["segment_ids"] = seg_ids
            sample["image_path"] = batch[3]

            if self.img_encoded:
                sample["img_embeddings"] = torch.stack(batch[4])
            else:
                sample["image"] = torch.stack(batch[4])

            return sample
        
        return collate_wrapper

    def __getitem__(self, index):
        sample = self.datasets[index]
        if self.img_encoded:
            img_emb = torch.from_numpy(np.load(sample["image_path"]))
            return sample["input_ids"], sample["labels"], sample["segment_ids"], sample["image_path"], img_emb
        else:
            img = Image.open(sample["image_path"]).convert("RGB")
            img = self.img_transform(img)
            return sample["input_ids"], sample["labels"], sample["segment_ids"], sample["image_path"], img
    
    def __len__(self):
        return len(self.datasets)

    def get_dataset(self, image_dir, anno, include_captioning=False):
        raise NotImplementedError

    def preprocess(self, img_path, question_ids, answer_ids, explanation_ids):
        raise NotImplementedError

    def get_relabel_targets(self):
        n_add = len(self.relabeled_ids_data)
        n_nle = len(self.nle_ids_data)
        
        if n_nle < n_add:
            sampled_indices = random.sample(list(range(n_add)), n_nle)
        else:
            sampled_indices = list(range(n_add))

        relabel_data = []
        for idx in sampled_indices:
            sample = self.relabeled_ids_data[idx].copy()
            sample["idx"] = idx
            relabel_data.append(sample)
        
        return relabel_data
    
    def renew_explanations(self, labels):
        for idx, sample in tqdm(labels.items(), desc="Renewal"):
            self.relabeled_ids_data[idx]["explain"] = sample

    def change_mode(self, mode):
        if (self.teacher_mode and mode=="teacher") or (not self.teacher_mode and mode=="student"):
            return
        if mode=="teacher":
            self.teacher_mode = True
        else:
            self.teacher_mode = False

        # cached_path = os.path.join(self.cfg.cached_dir, f"total_{mode}.cache")
        # if os.path.exists(cached_path):
        #     logger.info("Loading features from cached file %s", cached_path)
        #     self.datasets = torch.load(cached_path)
        # else:
        # Preprocessing dataset
        print("Before changing > NLE:", len(self.nle_ids_data), "Add:", len(self.relabeled_ids_data))
        
        if self.teacher_mode:
            n_nle = len(self.nle_ids_data)
            n_add = len(self.relabeled_ids_data)
            additional_data = random.sample(self.relabeled_ids_data, len(self.nle_ids_data)) if n_nle < n_add else self.relabeled_ids_data
            total_data = self.nle_ids_data + additional_data
        else:
            total_data = self.nle_ids_data + self.relabeled_ids_data

        datasets = []
        for sample in tqdm(total_data, desc=f"Changing to {mode} mode"):
            input_ids, segment_ids, labels = self.preprocess(
                sample["question"], sample["answer"], sample["explain"])
            datasets.append({"image_path": sample["image_path"], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids})
        
        self.datasets = datasets
        
        print("After changing > NLE:", len(self.nle_ids_data), "Add:", len(self.relabeled_ids_data))


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
            if self.img_encoded:
                img_path = os.path.join(image_dir+"_encoded", img_name+".npy")
            else:
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
        if self.teacher_mode:
            additional_data = random.sample(relabeled_data, len(nle_data))
            total_data = nle_data + additional_data
        else:
            total_data = nle_data + relabeled_data

        # Preprocessing dataset
        for sample in tqdm(total_data, desc="Tokenizing and Tensorizing"):
            input_ids, segment_ids, labels = self.preprocess(sample["question"], sample["answer"], sample["explain"])
            data.append({"image_path": sample["image_path"], "input_ids":input_ids, "labels": labels, "segment_ids":segment_ids})

        return data, nle_data, relabeled_data
    
    def preprocess(self, question_ids, answer_ids, explanation_ids):
        # Image    
        # img = Image.open(img_path).convert('RGB')

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
        # img = self.img_transform(img)
        
        return input_ids, segment_ids, labels