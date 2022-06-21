import os, json
import torch
from pytorch_lightning import Trainer, seed_everything
from dataloader import VQAXDataModule, QAGenDataset
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
    def __init__(self, metric="GreedyMatchingScore"):
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


def generate_QA_from_explanation(backbone, tokenizer, args):
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

        return sample

    args.inference = True
    model = PromptTuning.load_from_checkpoint(args.load_ckpt_path, strict=False, hparams=args, lm_backbone=backbone, tokenizer=tokenizer)
    trainer = Trainer(accelerator = "gpu", gpus=1)
    dataset = QAGenDataset(args.anno_path, args.object_label_path, tokenizer)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.n_workers,\
                            pin_memory=True, collate_fn=collate_wrapper)
    results = trainer.predict(model, dataloaders=dataloader)
    if not os.path.exists(args.qa_save_dir):
        os.mkdir(args.qa_save_dir)

    with open(f"{args.qa_save_dir}/origin_k-{args.top_k}_p-{args.top_p}_t-{args.temperature}.json", "w") as fout:
        json.dump(results, fout, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA')

    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument("--anno_path", type=str, required=True)
    data_args.add_argument("--object_label_path", type=str, required=True)
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
    model_args.add_argument("--qa_save_dir", type=str, default="aug_results")

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default=None, help='Experiment name for wandb')
    misc_args.add_argument('--ckpt_dir', type=str, default="./ckpts", help='Checkpoint directory')

    args = parser.parse_args()

    seed_everything(args.seed)

    # Load pretrained models
    config, tokenizer, backbone = load_T5_backbone(args)

    candidates = generate_QA_from_explanation(backbone, tokenizer, args=args)
