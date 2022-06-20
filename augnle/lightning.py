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
from transformers import SwinModel

BECAUSE = f"because"

class PromptTuning(LightningModule):
    def __init__(self, hparams, lm_backbone, image_encoder):
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

        self.model = lm_backbone
        self.image_encoder = image_encoder

        self.change_requires_grad(self.model, False)
        self.change_requires_grad(self.model.prefix_encoder, True)
    
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
        outputs = self.model(input_ids=batch["enc_inputs"], attention_mask=batch["attn_mask"], labels=batch["labels"])

        loss = outputs.loss

        self.log("prompt_train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["enc_inputs"], attention_mask=batch["attn_mask"], labels=batch["labels"])

        loss = outputs.loss

        loss = outputs.loss
        self.log("val_loss", loss)

        return loss
    
    def predict_step(self, batch, batch_idx: int):

        visual_embeddings = self.image_encoder(pixel_values=batch["img"]).last_hidden_state
        visual_embeddings = self.visual_proj(visual_embeddings)

        t_e_inputs = self.model.shared(batch["t_e_inputs"])
        t_e_inputs = torch.cat((visual_embeddings, t_e_inputs), dim=1)

        pseudo_label = self.model.generate(inputs_embeds=t_e_inputs, attention_mask=batch["t_e_attn_mask"], early_stopping= True)
        pseudo_label = pseudo_label.cpu().numpy().tolist()[0]
        pseudo_label = pseudo_label[1:-1]
        pseudo_label = self.tokenizer.decode(pseudo_label, clean_up_tokenization_spaces=True)
        
        return pseudo_label
    
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

    def get_trained_weights(self):
        return self.model, self.image_encoder


class Adaptation(LightningModule):
    def __init__(self, hparams, lm_backbone, image_encoder, tokenizer):
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

        self.discrete_prompt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.cfg.discrete_prompt))
        self.discrete_prompt = torch.tensor(self.discrete_prompt, dtype=torch.long).unsqueeze(0)

        self.model = lm_backbone
        self.image_encoder = image_encoder
        self.model.unfreeze()
    
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
        visual_embeddings = self.image_encoder(pixel_values=batch["img"])

        enc_inputs = self.model.shared(batch["enc_inputs"])
        prompt = self.discrete_prompt.expand(enc_inputs.size(0),-1)
        prompt = self.model.shared(prompt.to(enc_inputs.get_device()))
        enc_inputs = torch.cat((prompt, visual_embeddings, enc_inputs), dim=1)

        outputs = self.model(inputs_embeds=enc_inputs, \
                            attention_mask=batch["enc_attn_mask"], \
                            labels=batch["label"], \
                            prompting=False)

        loss = outputs.loss

        self.log("adapt_train_loss", loss)

        return loss
    


    def validation_step(self, batch, batch_idx):
        visual_embeddings = self.image_encoder(pixel_values=batch["img"])

        enc_inputs = self.model.shared(batch["enc_inputs"])
        prompt = self.discrete_prompt.expand(enc_inputs.size(0),-1)
        prompt = self.model.shared(prompt.to(enc_inputs.get_device()))
        enc_inputs = torch.cat((prompt, visual_embeddings, enc_inputs), dim=1)

        outputs = self.model(inputs_embeds=enc_inputs, \
                            attention_mask=batch["enc_attn_mask"], \
                            labels=batch["label"], \
                            prompting=False)

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

    def get_trained_weights(self):
        return self.model, self.image_encoder