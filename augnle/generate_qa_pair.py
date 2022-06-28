import os, json
import torch
from pytorch_lightning import Trainer, seed_everything
from dataloader import VQAXDataModule, QAGenDataset, AnswerGenDataset, QuestionRefineDataset
from models.lightning import PromptTuning
import argparse
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Config, T5Tokenizer
from models.seq2seq import T5PrefixForConditionalGeneration
from torch.utils.data import DataLoader
from nlgeval import NLGEval
# from models.ViT import SwinImageEncoder

class MatchingScorer(object):
    """
    scorer = MatchingScorer(metric="EmbeddingAverageCosineSimilarity")
    print(scorer.forward("banana", "only banana"))
    """
    def __init__(self, metric="EmbeddingAverageCosineSimilarity"):
        matching_metrics = [
            'Bleu_1',
            'EmbeddingAverageCosineSimilarity',
            'VectorExtremaCosineSimilarity',
            'GreedyMatchingScore', 
            'METEOR', 'ROUGE_L', 'CIDEr', 'SkipThoughtCS']

        if "Bleu_" == metric[:-1]:
            except_metrics = matching_metrics[1:]
        else:
            idx = matching_metrics.index(metric)
            except_metrics = matching_metrics[:idx] + matching_metrics[idx+1:]
        self.metric = metric
        self.evaluator = NLGEval(metrics_to_omit=except_metrics)
    
    def forward(self, hyp, ref):
        if type(ref) is str:
            ref = [ref]
        scores = self.evaluator.compute_individual_metrics(hyp=hyp, ref=ref)
        return scores[self.metric]


AVAIL_GPUS = torch.cuda.device_count()

def load_T5_backbone(args):
    #Configuration
    config = T5Config.from_pretrained(args.lm_backbone)
    setattr(config, 'prefix_dropout', args.prefix_dropout)
    setattr(config, 'prefix_projection', args.prompt_proj)
    setattr(config, 'prefix_len', args.prefix_len)

    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.lm_backbone)
    
    # backbone
    backbone = T5PrefixForConditionalGeneration.from_pretrained(args.lm_backbone, config=config, tokenizer=tokenizer)
    #backbone.resize_token_embeddings(len(tokenizer))

    return config, tokenizer, backbone

def decode_to_QAE(results):
    decoded = []
    fails = []
    n_fail_cases = 0
    for result in results:
        reason = result["input"].split(".")
        try:
            obj_label = reason[1]
            if len(reason) > 2:
                reason = result["input"].split(". there")
                obj_label = "there" + reason[1]
            reason = reason[0]

            if type(result["output"]) is not list:
                result["output"] = [result["output"]]
                
            for out in result["output"]:
                question, answer = out.split("? the answer is ")
                decoded.append({
                    "question": question,
                    "answer": answer,
                    "reason": reason,
                    "img_name": result["img_name"],
                    "objects": obj_label.strip()
                })
        except Exception as e:
            n_fail_cases += 1
            fails.append(result)

    print("The number of failure cases:", n_fail_cases)

    return decoded, fails


def collate_wrapper_no_label(batch):
    batch = list(zip(*batch))
    sample = {}

    # enc max len
    enc_max_len = max([x.size(0) for x in batch[0]])
    # enc_input
    enc_inputs = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
    enc_attn_mask = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
    for i, x in enumerate(batch[0]):
        enc_inputs[i,:x.size(0)] = x
        enc_attn_mask[i,:x.size(0)] = 1.0
    
    sample["enc_inputs"] = enc_inputs
    sample["attn_mask"] = enc_attn_mask
    sample["img_names"] = [x for x in batch[1]]

    return sample
    

def generate_QA_from_explanation(backbone, tokenizer, args):
    args.inference = True
    model = PromptTuning.load_from_checkpoint(args.load_ckpt_path, strict=False, hparams=args, lm_backbone=backbone, tokenizer=tokenizer)
    model.set_prompt_A()
    trainer = Trainer(accelerator = "gpu", gpus=1)
    dataset = QAGenDataset(args.object_label_paths, tokenizer)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.n_workers,\
                            pin_memory=True, collate_fn=collate_wrapper_no_label)
    results = trainer.predict(model, dataloaders=dataloader)
    
    candidates = []
    for b in results:
        candidates.extend(b)
    
    candidates, fails = decode_to_QAE(candidates)

    if not os.path.exists(args.qa_save_dir):
        os.mkdir(args.qa_save_dir)

    with open(f"{args.qa_save_dir}/origin_k-{args.top_k}_p-{args.top_p}_t-{args.temperature}.json", "w") as fout:
        json.dump(candidates, fout, indent=2)

    with open(f"{args.qa_save_dir}/fail.json", "w") as fout:
        json.dump(fails, fout, indent=2)

    return candidates, model


def is_matched(output, label):
    if output==label:
        return True
    
    output = output.split("the answer is ")[-1]
    label = label.split("the answer is ")[-1]

    out_no_space = "".join(output.split(" "))
    label_no_space = "".join(label.split(" "))
    if out_no_space==label_no_space:
        return True

    if len(output) < 4 and len(label) < 4:
        return False

    if len(output) > len(label) and (output[:-2]==label or output[:-1]==label):
        return True
    
    if len(output) < len(label) and (output==label[:-2] or output==label[:-1]):
        return True
    
    return False

def answer_based_filtering(candidates, model, tokenizer, args):
    def collate_wrapper(batch):
        batch = list(zip(*batch))
        sample = {}

        # enc max len
        enc_max_len = max([x.size(0) for x in batch[0]])
        # enc_input
        enc_inputs = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
        enc_attn_mask = torch.zeros((len(batch[0]), enc_max_len), dtype=torch.long)
        for i, x in enumerate(batch[0]):
            enc_inputs[i,:x.size(0)] = x
            enc_attn_mask[i,:x.size(0)] = 1.0
        
        # label
        dec_max_len = max([x.size(0) for x in batch[1]])
        label = torch.zeros((len(batch[1]), dec_max_len), dtype=torch.long)
        for i, x in enumerate(batch[1]):
            label[i,:x.size(0)] = x

        sample["enc_inputs"] = enc_inputs
        sample["attn_mask"] = enc_attn_mask
        sample["labels"] = label
        sample["img_names"] = [x for x in batch[2]]
        sample["objects"] = [x for x in batch[3]]

        return sample

    name = f"k-{args.top_k}_p-{args.top_p}_t-{args.temperature}_n_{args.num_return_sequences}.json"
    args.inference = True
    args.temperature = 1.0
    args.num_return_sequences = 1
    args.top_k = 0
    args.top_p = 0.95
    model.set_prompt_B()

    trainer = Trainer(accelerator="gpu", gpus=1)
    dataset = AnswerGenDataset(candidates, tokenizer)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.n_workers,\
                            pin_memory=True, collate_fn=collate_wrapper)
    results = trainer.predict(model, dataloaders=dataloader)

    pair_data = []
    for b in results:
        pair_data.extend(b)

    matched = []
    unmatched = []
    for data in pair_data:
        if is_matched(data["output"], data["label"]):
            question, reason = data["input"].split("? reason: ")
            matched.append({
                "question": question,
                "reason": reason,
                "answer": data["output"],
                "img_name": data["img_name"],
                "objects": data["objects"]
            })
        else:
            unmatched.append(data)
    
    match_rate = "%.2f" % ((len(matched)/len(pair_data))*100)
    print("Match rate: {}".format(match_rate))
    
    with open(args.qa_save_dir + "/filtered_"+match_rate+"_"+name, "w") as fout:
        json.dump(matched, fout, indent=2)
    
    with open(args.qa_save_dir + "/wrong_"+match_rate+"_"+name, "w") as fout:
        json.dump(unmatched, fout, indent=2)
    
    return matched, unmatched


def decode_input_to_QAE(text):
    q, text = text.split("? ")
    e_idx = text.find('because')
    a = text[:e_idx]
    text = text[e_idx:]
    text = text.split(".")
    if len(text)==2:
        reason, objects = text
    else:
        reason = text[0]
        objects = ""

    return {"question": q, "answer": a.strip(), "reason": reason.strip(), "objects":objects.strip()}


def refine_question(matched, model, tokenizer, args):
    name = f"k-{args.top_k}_p-{args.top_p}_t-{args.temperature}_n_{args.num_return_sequences}.json"
    args.inference = True
    model.set_prompt_C()

    trainer = Trainer(accelerator="gpu", gpus=1)
    dataset = QuestionRefineDataset(matched, tokenizer)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.n_workers,\
                            pin_memory=True, collate_fn=collate_wrapper_no_label)
    results = trainer.predict(model, dataloaders=dataloader)

    pair_data = []
    for b in results:
        pair_data.extend(b)

    refined_samples = []
    n_accepted = 0
    for data in pair_data:
        decoded = decode_input_to_QAE(data["input"])
        decoded.update({"img_name": data["img_name"]})
        if data["output"]=="acceptable":
            refined_samples.append(decoded)
            n_accepted += 1
        else:
            try:
                refined_question = data["output"].split("not acceptable question: ")[-1]
                decoded["ori_question"] = decoded["question"]
                decoded["question"] = refined_question
                refined_samples.append(decoded)
            except Exception as e:
                print(e)
                
    acceptance_rate = "%.2f" % ((n_accepted/len(pair_data))*100)
    print("Acceptance rate: {}".format(acceptance_rate))
    
    with open(args.qa_save_dir + "/refined_"+acceptance_rate+"_"+name, "w") as fout:
        json.dump(refined_samples, fout, indent=2)
    
    return refined_samples
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA')

    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    # data_args.add_argument("--anno_path", type=str, required=True)
    data_args.add_argument("--object_label_paths", nargs="+", required=True)
    data_args.add_argument("--batch_size", type=int, default=32)
    data_args.add_argument("--n_workers", type=int, default=8)
    
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument("--lm_backbone", type=str, default="google/t5-xl-lm-adapt", help="Pretrained language model")
    model_args.add_argument("--prompt_proj", action="store_true", help="Project prefix")
    model_args.add_argument("--prefix_len", type=int, default=100)
    model_args.add_argument("--prefix_dropout", type=float, default=0.1)

    """Inference related arguments"""
    model_args.add_argument("--load_ckpt_path", required=True, type=str)
    model_args.add_argument("--top_p", type=float, default=None)
    model_args.add_argument("--top_k", type=int, default=None)
    model_args.add_argument("--temperature",type=float, default=1.0, help=" Temperature ")
    model_args.add_argument("--num_return_sequences", type=int, default=1)
    model_args.add_argument("--qa_save_dir", type=str, default="aug_results")

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default=None, help='Experiment name for wandb')
    misc_args.add_argument('--ckpt_dir', type=str, default="./ckpts", help='Checkpoint directory')

    args = parser.parse_args()

    seed_everything(args.seed)
    args.inference = True

    # Load pretrained models
    config, tokenizer, backbone = load_T5_backbone(args)

    candidates, model = generate_QA_from_explanation(backbone, tokenizer, args=args)
    matched, unmatched = answer_based_filtering(candidates, model, tokenizer, args)

    # model = PromptTuning.load_from_checkpoint(args.load_ckpt_path, strict=False, hparams=args, lm_backbone=backbone, tokenizer=tokenizer)
    # with open("../captioning_data/nocaps/annotations/filtered_80.34_k-100_p-0.5_t-0.7_n_1.json", "r") as fin:
    #     matched = json.load(fin)
    # refine_question(matched, model, tokenizer, args)