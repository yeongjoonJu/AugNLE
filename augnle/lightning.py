import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import (
    get_linear_schedule_with_warmup,
     T5Config,
     T5Tokenizer,
)
from models.seq2seq import T5PrefixForConditionalGeneration
from models.ViT import ImageEncoder
from transformers import SwinModel


class PromptTuning(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        # Save Hyper parameters
        self.save_hyperparameters(hparams)
        self.cfg = hparams
        self.learning_rate= self.cfg.learning_rate
        self.adam_epsilon=  self.cfg.adam_epsilon
        self.warmup_steps=  self.cfg.warmup_steps
        self.weight_decay=  self.cfg.weight_decay
        self.img_size = self.cfg.img_size
        self.max_seq_length = self.hparams.enc_max_len
        

        #Configuration
        self.config = T5Config.from_pretrained(hparams.lm_backbone)
        setattr(self.config, 'img_size', hparams.img_size)
        setattr(self.config, 'max_seq_length', hparams.enc_max_len)
        setattr(self.config, 'prefix_dropout', hparams.prefix_dropout)
        setattr(self.config, 'prefix_projection', not hparams.no_prompt_proj)
        setattr(self.config, 'prefix_hidden_size', hparams.prefix_hidden_size)
        setattr(self.config, 'prefix_len', hparams.prefix_len)
        
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.lm_backbone)
        # num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<situation>', '<answer>']})
        
        # self.config.add_cross_attention = True
        self.model = T5PrefixForConditionalGeneration.from_pretrained(hparams.lm_backbone, config=self.config, tokenizer=self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # self.image_encoder = ImageEncoder(self.device)
        self.image_encoder = SwinModel.from_pretrained(hparams.visual_backbone)
        self.visual_proj = nn.Sequential(
            nn.Linear(self.image_encoder.num_features, self.image_encoder.num_features),
            nn.GELU(),
            nn.Linear(self.image_encoder.num_features, self.config.d_model)
        )

        # Freezing
        if self.hparams.finetuning:
            self.change_requires_grad(self.model.prefix_encoder, False)
    
        self.change_requires_grad(self.image_encoder, False)
        
    def change_requires_grad(self, model, req_grad):
        for p in model.parameters():
            p.requires_grad = req_grad

    def setup(self,stage):
        train_iter = len(self.trainer.datamodule.train_dataloader())
        
        # Setting
        tb_size = self.hparams.train_batch_size  * self.trainer.accumulate_grad_batches * max(1, self.trainer.gpus)
        self.total_steps = (train_iter // tb_size) * self.trainer.max_epochs
        self.warmup_steps = int(train_iter / self.trainer.gpus * self.trainer.max_epochs * 0.2)
    

    def forward(self, **inputs):
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        visual_embeddings = self.image_encoder(pixel_values=batch["img"]).last_hidden_state
        visual_embeddings = self.visual_proj(visual_embeddings)

        t_e_inputs = self.model.shared(batch["t_e_inputs"])
        t_e_inputs = torch.cat((visual_embeddings, t_e_inputs), dim=1)
        t_a_inputs = self.model.shared(batch["t_a_inputs"])

        enc_inputs = torch.cat((t_e_inputs, t_a_inputs), dim=0)
        attn_mask = torch.cat((batch["t_e_attn_mask"], batch["t_a_attn_mask"]), dim=0)
        labels = torch.cat((batch["t_e_label"], batch["t_a_label"]), dim=0)

        outputs = self.model(inputs_embeds=enc_inputs, attention_mask=attn_mask, labels=labels)

        loss = outputs.loss

        self.log("train_loss", loss)

        return loss


    def validation_step(self, batch, batch_idx):
        visual_embeddings = self.image_encoder(pixel_values=batch["img"]).last_hidden_state
        visual_embeddings = self.visual_proj(visual_embeddings)

        t_e_inputs = self.model.shared(batch["t_e_inputs"])
        t_e_inputs = torch.cat((visual_embeddings, t_e_inputs), dim=1)
        t_a_inputs = self.model.shared(batch["t_a_inputs"])

        enc_inputs = torch.cat((t_e_inputs, t_a_inputs), dim=0)
        attn_mask = torch.cat((batch["t_e_attn_mask"], batch["t_a_attn_mask"]), dim=0)
        labels = torch.cat((batch["t_e_label"], batch["t_a_label"]), dim=0)

        outputs = self.model(inputs_embeds=enc_inputs, attention_mask=attn_mask, labels=labels)

        loss = outputs.loss
        self.log("val_loss", loss)

        return loss


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]