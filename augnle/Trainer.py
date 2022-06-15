import os
import torch
from pytorch_lightning import Trainer, seed_everything
from dataloader import VQAXDataModule
from lightning import Adaptation, PromptTuning
import opts
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Config, T5Tokenizer
from models.seq2seq import T5PrefixForConditionalGeneration
from models.ViT import SwinImageEncoder

AVAIL_GPUS = torch.cuda.device_count()

def load_T5_backbone(args):
    #Configuration
    config = T5Config.from_pretrained(args.lm_backbone)
    setattr(config, 'img_size', args.img_size)
    setattr(config, 'max_seq_length', args.enc_max_len)
    setattr(config, 'prefix_dropout', args.prefix_dropout)
    setattr(config, 'prefix_projection', not args.no_prompt_proj)
    setattr(config, 'prefix_hidden_size', args.prefix_hidden_size)
    setattr(config, 'prefix_len', args.prefix_len)

    # Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.lm_backbone)
    
    # backbone
    backbone = T5PrefixForConditionalGeneration.from_pretrained(args.lm_backbone, config=config, tokenizer=tokenizer)
    #backbone.resize_token_embeddings(len(tokenizer))

    return config, tokenizer, backbone


def multi_task_prompt_tuning(backbone, image_encoder, args):
    args.filename = 'ckpt_stats_' + str(args.load_from_epoch) + '.tar'
    
    # Checkpoint call back
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d_%H:%M')
    ckpt_dir = args.ckpt_dir + '/' + nowDatetime + "/"
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    wandb_logger = WandbLogger(project="Aug_NLX", name=args.experiment_name)
    checkpoint_callback = checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode = "min",
        save_top_k = 2,
        dirpath=ckpt_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
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

    dm = VQAXDataModule(args, mode="prompt")
    model = PromptTuning(hparams=args, lm_backbone=backbone, image_encoder=image_encoder)
    trainer.fit(model, datamodule=dm)

    return model.get_trained_weights()


def adaptation(backbone, image_encoder, tokenizer, args):
    args.filename = 'ckpt_stats_' + str(args.load_from_epoch) + '.tar'
    
    # Checkpoint call back
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d_%H:%M')
    ckpt_dir = args.ckpt_dir + '/' + nowDatetime + "/"
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
        
    wandb_logger = WandbLogger(project="Aug_NLX", name=args.experiment_name)
    checkpoint_callback = checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode = "min",
        save_top_k = 2,
        dirpath=ckpt_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
    )
    trainer = Trainer(max_epochs = 3,
                    accelerator = "gpu",
                    gpus= args.ngpu,
                    strategy = "ddp",
                    val_check_interval=args.val_check_interval,
                    accumulate_grad_batches = args.gradient_accumulation_steps,
                    gradient_clip_val=args.gradient_cliping,
                    check_val_every_n_epoch = 1,
                    callbacks=[checkpoint_callback],
                    logger=wandb_logger,)

    dm = VQAXDataModule(args, mode="adapt")
    model = Adaptation(hparams=args, lm_backbone=backbone, image_encoder=image_encoder, tokenizer=tokenizer)
    trainer.fit(model, datamodule=dm)

    return model.get_trained_weights()


if __name__ == '__main__':
    args = opts.get_args()
    seed_everything(args.seed)

    # Load pretrained models
    config, tokenizer, backbone = load_T5_backbone(args)
    image_encoder = SwinImageEncoder(args.visual_backbone, config.d_model)

    # Adaptation
    backbone, image_encoder = adaptation(backbone, image_encoder, tokenizer, args)

    backbone.cpu()
    image_encoder.cpu()
    torch.cuda.empty_cache()

    # Multi-task prompt tuning
    multi_task_prompt_tuning(backbone, image_encoder, args)
