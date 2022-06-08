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

class PromptTuning(LightningModule):
    def __init__(
        self,
        hparams,
        **kwargs,
    ):
        super().__init__()
        # Save Hyper parameters
        self.save_hyperparameters(hparams)
        # version difference
        #self.hparams.update(hparams)
        self.task_name = "task_A" if self.hparams.task_A else "task_B"
        self.learning_rate= self.hparams.learning_rate
        self.adam_epsilon= self.hparams.adam_epsilon
        self.warmup_steps= self.hparams.warmup_steps
        self.weight_decay= self.hparams.weight_decay
        self.train_batch_size= self.hparams.train_batch_size
        self.eval_batch_size= self.hparams.eval_batch_size
        self.ckpt_path = self.hparams.ckpt_path
        self.img_size = self.hparams.img_size
        self.max_seq_length = self.hparams.input_max_seq_length
        

        #Configuration
        self.config = T5Config.from_pretrained('t5-large')
        setattr(self.config, 'img_size', None)
        setattr(self.config, 'task_name', self.task_name)
        setattr(self.config, 'max_seq_length', None)
        setattr(self.config, 'hidden_dropout_prob', 0.1)
        setattr(self.config, 'prefix_projection', True)
        setattr(self.config, 'prefix_hidden_size', 40)
        setattr(self.config, 'prefix_seq_len', 20)
        
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<scene>', '<answer>']})
        
        self.config.img_size = self.img_size
        self.config.max_seq_length = self.max_seq_length 
        self.config.add_cross_attention = True
        self.model = T5PrefixForConditionalGeneration.from_pretrained('t5-large',config = self.config, tokenizer = self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.image_encoder = ImageEncoder(self.device)
        
        self.mlp_vit = nn.Sequential(
            nn.Linear(self.image_encoder.encoder.visual.conv1.out_channels, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.d_model)
        )
        
        # Freezing
        if self.hparams.finetuning:
            self.change_requires_grad(self.model.prefix_encoder, False)
    
        self.change_requires_grad(self.image_encoder, False)
        

        
    def change_requires_grad(self, model, req_grad):
        for p in model.parameters():
            p.requires_grad = req_grad
    
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self,  batch, batch_idx):
        
        if self.task_name == "task_A":
            p_id, img, _, input_ids, labels, _, attention_mask = batch.values()
            img_embeddings = self.image_encoder(img)
            img_embeddings = self.mlp_vit(img_embeddings)
        
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=None,
                inputs_embeds=img_embeddings,
                labels=labels,
                )
        # Task B
        else:
            p_id, _, _, input_ids, labels, _, attention_mask = batch.values()
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=None,
                inputs_embeds=None,
                labels=labels,
                )
        loss = outputs.loss
        
        
        self.log("train_loss", loss)

        return loss
    def setup(self,stage):
        train_loader = self.trainer.datamodule.train_dataloader()
        
        # Setting
        tb_size = self.hparams.train_batch_size  * self.trainer.accumulate_grad_batches * max(1, self.trainer.gpus)
        self.total_steps = (len(train_loader.dataset) // tb_size) * self.trainer.max_epochs
        self.warmup_steps = int(len(train_loader.dataset) / self.trainer.gpus * self.trainer.max_epochs * 0.2)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.task_name == "task_A":
            p_id, img, _, input_ids, labels, _, attention_mask = batch.values()
            img_embeddings = self.image_encoder(img)
            img_embeddings = self.mlp_vit(img_embeddings)
        
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=None,
                inputs_embeds=img_embeddings,
                labels=labels,
                )
        # Task A
        else:
            p_id, _, _, input_ids, labels, _, attention_mask = batch.values()
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=None,
                inputs_embeds=None,
                labels=labels,
                
                )
        loss = outputs.loss
        self.log("val_loss", loss)
        return outputs[0]


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

'''
    
    def pred_file(self,prefix=''):
        self.output_prediction_file = os.path.join(self.hparams.output_dir, "predictions_{}.json".format(prefix))
        self.output_nbest_file = os.path.join(self.hparams.output_dir, "nbest_predictions_{}.json".format(prefix))
        if self.hparams.version_2_with_negative:
            self.output_null_log_odds_file = os.path.join(self.hparams.output_dir, "null_odds_{}.json".format(prefix))
        else:
            self.output_null_log_odds_file = None
            '''