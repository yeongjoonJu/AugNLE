import os
from transformers import GPT2Tokenizer
import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from dataloader import VQAX_full_shot_Dataset, VQAXEvalDataset
from models.lightning import Self_training, NLX_GPT
import opts
import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import json
# from models.utils import pseudo_labeling

import logging
logger = logging.getLogger(__name__)

AVAIL_GPUS = torch.cuda.device_count()

class BaseTrainer(object):
    def __init__(self, project_name, include_captioning, args):
        self.args = args
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        orig_num_tokens = len(tokenizer.encoder)
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        assert len(tokenizer) == orig_num_tokens + num_new_tokens

        if args.mode=="test":
            self.trainer = Trainer(accelerator="gpu", gpus=1)
            self.test_dataset = VQAXEvalDataset(
                path=args.nle_test_anno_path,
                img_dir=args.nle_test_image_dir,
                tokenizer=tokenizer,
                args=args)
        else:
            assert args.fewshot_ratio < 1.0
            if args.fewshot_ratio > 0 or args.fewshot_num is not None or args.external_data:
                self.fewshot = True
                pass
            else:
                self.fewshot = False
                self.train_dataset = VQAX_full_shot_Dataset(args.nle_train_image_dir, args.nle_train_anno_path, tokenizer, args, include_captioning=include_captioning)
                self.valid_dataset = VQAX_full_shot_Dataset(args.nle_valid_image_dir, args.nle_valid_anno_path, tokenizer, args, include_captioning=False)

            # Define checkpoint callback
            ckpt_callback = self.get_checkpoint_callback()
            # Define logger
            logger = WandbLogger(project=project_name, name=args.experiment_name)
            # Define trainer
            self.trainer = Trainer(max_epochs=args.max_epochs,
                                accelerator = "gpu",
                                gpus= args.ngpu,
                                strategy = "ddp",
                                val_check_interval=args.val_check_interval,
                                accumulate_grad_batches = args.gradient_accumulation_steps,
                                gradient_clip_val=args.gradient_cliping,
                                check_val_every_n_epoch = 1,
                                callbacks=[ckpt_callback],
                                logger=logger,)

        self.tokenizer = tokenizer


    def get_checkpoint_callback(self):
        # Checkpoint call back
        now = datetime.datetime.now()
        if self.args.experiment_name is None:
            nowDatetime = now.strftime('%Y-%m-%d_%H:%M')
            ckpt_dir = self.args.checkpoints_dir + '/' + nowDatetime + "/"
        else:
            ckpt_dir = self.args.checkpoints_dir + "/" + self.args.experiment_name

        checkpoint_callback = checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode = "min",
            save_top_k = 2,
            dirpath=ckpt_dir,
            filename='{epoch:02d}-{val_loss:.3f}',
        )

        return checkpoint_callback

    def get_dataloaders(self):
        collate_fn = self.train_dataset.get_collate_fn()
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.args.train_batch_size, \
                                    pin_memory=True, num_workers=self.args.n_train_workers, collate_fn=collate_fn)
        valid_dataloader = DataLoader(self.valid_dataset, shuffle=False, batch_size=self.args.train_batch_size, \
                                    pin_memory=True, num_workers=self.args.n_valid_workers, collate_fn=collate_fn)

        return train_dataloader, valid_dataloader
    
    def fit(self):
        train_loader, valid_loader = self.get_dataloaders()
        self.args.total_steps = len(train_loader) // self.args.gradient_accumulation_steps // self.args.ngpu * self.args.max_epochs
        if self.args.warmup_steps < 0:
            self.args.warmup_steps = self.args.total_steps * self.args.warmup_ratio
        model = NLX_GPT(self.tokenizer, hparams=args)
        self.trainer.fit(model, train_loader, valid_loader)

    def test(self):
        model = NLX_GPT.load_from_checkpoint(self.args.load_ckpt_path, strict=False, tokenizer=self.tokenizer, hparams=self.args)
        dataloader = DataLoader(self.test_dataset, shuffle=False, batch_size=1, num_workers=self.args.n_valid_workers)
        self.trainer.test(model, dataloader)


class Self_E(BaseTrainer):
    """
    experiment 1 (Few-shot setting)
    NLE data (few-shot rate) / Captioning data / unlabeled NLE data (1. - few-shot rate)

    experiment 2 (Full-shot setting)
    NLE data (100%) / Captioning data

    experiment 3 (Domain Adaptation setting)
    NLE data (100%) / Captioning data / Target domain dataset
    """
    def __init__(self, args):
        super().__init__("Self-E", True, args)

    def teacher_training(self):
        train_loader, valid_loader = self.get_dataloaders()
        self.args.total_steps = len(train_loader) // self.args.gradient_accumulation_steps // self.args.ngpu * self.args.max_epochs
        if self.args.warmup_steps < 0:
            self.args.warmup_steps = self.args.total_steps * self.args.warmup_ratio
        model = Self_training(hparams=args)
        self.trainer.fit(model, train_loader, valid_loader)

        return model

    def student_training(self):
        pass

    def generate_explanation(self):
        predictor = Trainer(accelerator="gpu", gpus=1)
        pass
    
    def fit(self):
        for it in range(self.args.max_iter):
            # teacher training
            print(f"=== Self-E Training: {it+1} iteration - Teacher training ===")
            model = self.teacher_training()

            # Generating explanations
            if self.fewshot:
                pass
            else:
                pass

            # E = self.trainer.fit(teacher, I, Q, A) # <= NLE data / captioning data
            # # explanation generation using teacher model
            # E = self.trainer.predict(teacher, I, Q, A) # <= unlabeled and captioning data (part1)
            # # student training
            # A, E = self.trainer.fit(student, I, Q) # <= captioning data / NLE data / pseudo-labeled
            # # answer and explanation generation using student model
            # A, E = self.trainer.predict(student, I, Q) # <= captioning data (part2)
            # # check convergence
            # self.trainer.test(student)
            # teacher = student

        
if __name__ == '__main__':
    seed_everything(42)
    args = opts.get_args()

    trainer = BaseTrainer("nlx_gpt", include_captioning=False, args=args)
    if args.mode=="train":
        trainer.fit()
    else:
        trainer.test()

    # self_e = Self_E(args)
    # self_e.fit()