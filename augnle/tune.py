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

# from models.ViT import SwinImageEncoder

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


def multi_task_prompt_tuning(backbone, tokenizer, args):
    # Checkpoint call back
    now = datetime.datetime.now()
    if args.experiment_name is None:
        nowDatetime = now.strftime('%Y-%m-%d_%H:%M')
        ckpt_dir = args.ckpt_dir + '/' + nowDatetime + "/"
    else:
        ckpt_dir = args.ckpt_dir + "/" + args.experiment_name

    wandb_logger = WandbLogger(project="Aug_NLX", name=args.experiment_name)
    checkpoint_callback = checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode = "min",
        save_top_k = 2,
        dirpath=ckpt_dir,
        filename='{epoch:02d}-{val_loss:.3f}',
    )
    trainer = Trainer(max_epochs=args.max_epochs,
                    accelerator = "gpu",
                    gpus= args.ngpu,
                    strategy = "ddp",
                    val_check_interval=args.val_check_interval,
                    accumulate_grad_batches = args.gradient_accumulation_steps,
                    gradient_clip_val=args.gradient_cliping,
                    check_val_every_n_epoch = 1,
                    callbacks=[checkpoint_callback],
                    logger=wandb_logger,)

    dm = VQAXDataModule(args)
    args.inference = False
    model = PromptTuning(hparams=args, lm_backbone=backbone, tokenizer=tokenizer)

    class_A = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("question"))
    class_B = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("answer"))
    model.class_label_initialization(class_A, class_B)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA')

    """Optimization related arguments"""
    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--train_batch_size', type=int,  default= 32, help='Training batch Size')
    optim_args.add_argument('--eval_batch_size', type=int,  default= 32, help='Evalutatino batch Size')
    optim_args.add_argument('--adam_epsilon', type=float,  default= 1e-8, help='Adam epsilon')
    optim_args.add_argument('--warmup_steps', type=int,  default= 100, help='Warmup Steps')
    optim_args.add_argument('--weight_decay', type=float,  default= 0.04, help='Warmup Steps')
    optim_args.add_argument('--learning_rate', type=float,  default=5e-5, help='Initial Learning rate')
    optim_args.add_argument( "--gradient_accumulation_steps",type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    optim_args.add_argument( "--val_check_interval",type=float, default=0.5, 
                            help="validation check interval ratio")
    optim_args.add_argument( "--gradient_cliping",type=float, default=1.0, 
                            help=" The value at which to clip gradients ")
    optim_args.add_argument( "--temperature",type=int, default=0, help=" Temperature ")
    optim_args.add_argument('--requires_grad', action="store_true", help='requiring gradients')
    optim_args.add_argument('--num_workers', type=int, default=1, help='The number of workers')
    
    """Data related arguments"""
    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--train_anno_path', type=str, default=None, help='Path to annotation for training')
    data_args.add_argument('--valid_anno_path', type=str, default=None, help='Path to annotation for validation')
    data_args.add_argument("--fewshot_ratio", type=float, default=-1, help="Ratio of few-shot data")
    data_args.add_argument("--fewshot_num", type=int, default=500, help="The number of few-shot data")
    data_args.add_argument("--cached_dir", type=str, default="cached", help="Directory with cached file")
    data_args.add_argument("--n_train_workers", type=int, default=8)
    data_args.add_argument("--n_valid_workers", type=int, default=4)
    data_args.add_argument("--discrete_prompt", type=str, default="answer and explain: ")
    data_args.add_argument("--object_label_dir", type=str, default="../nle_anno/VQA-X")
    
    """Model related arguments"""
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument("--lm_backbone", type=str, default="google/t5-xl-lm-adapt", help="Pretrained language model")
    model_args.add_argument('--max_epochs', type=int, default=10, help='Max epoch size')
    model_args.add_argument('--load_from_epoch', type=str, default=None, help='Loading from epoch')
    model_args.add_argument("--prompt_proj", action="store_true", help="Project prefix")
    model_args.add_argument("--prefix_len", type=int, default=100)
    model_args.add_argument("--prefix_dropout", type=float, default=0.1)

    """Logging related arguments"""
    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=42, help='Random Seed')
    misc_args.add_argument('--experiment_name', type=str, default='experiment', help='Experiment name for wandb')
    misc_args.add_argument('--ngpu', type=int, default=1, help='Number of gpu')
    misc_args.add_argument('--ckpt_dir', type=str, default="./ckpts", help='Checkpoint directory')




    args = parser.parse_args()

    seed_everything(args.seed)

    # Load pretrained models
    config, tokenizer, backbone = load_T5_backbone(args)

    # Multi-task prompt tuning
    multi_task_prompt_tuning(backbone, tokenizer, args)
