import os
from unittest.util import _MAX_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import (
    get_linear_schedule_with_warmup,
     T5Config,
     T5Tokenizer,
)
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput


class PromptTuning(LightningModule):
    def __init__(self, hparams, lm_backbone, tokenizer):
        super().__init__()
        self.cfg = hparams
        if not hparams.inference:
            # Save Hyper parameters
            self.save_hyperparameters(hparams)
            self.learning_rate= self.cfg.learning_rate
            self.adam_epsilon=  self.cfg.adam_epsilon
            self.warmup_steps=  self.cfg.warmup_steps
            self.weight_decay=  self.cfg.weight_decay

        self.eos_token_id = tokenizer.eos_token_id
        
        self.tokenizer = tokenizer
        self.model = lm_backbone

        self.change_requires_grad(self.model, False)
        self.change_requires_grad(self.model.prefix_encoder_A, True)
        self.change_requires_grad(self.model.prefix_encoder_B, True)
    
    def class_label_initialization(self, class_idx_A, class_idx_B):
        class_idx_A = torch.LongTensor(class_idx_A)
        class_idx_B = torch.LongTensor(class_idx_B)
        self.model.class_label_initialization(class_idx_A, class_idx_B)
    
    def change_requires_grad(self, model, req_grad):
        for p in model.parameters():
            p.requires_grad = req_grad

    def setup(self,stage):
        if self.trainer.datamodule is not None:
            train_iter = len(self.trainer.datamodule.train_dataloader())
            
            # Setting
            tb_size = self.hparams.train_batch_size  * self.trainer.accumulate_grad_batches * max(1, self.trainer.gpus)
            self.total_steps = (train_iter // tb_size) * self.trainer.max_epochs
            if self.cfg.warmup_steps < 0:
                self.warmup_steps = int(train_iter / self.trainer.gpus * self.trainer.max_epochs * 0.2)
    

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["enc_inputs"], attention_mask=batch["attn_mask"], \
                            labels=batch["labels"], prompting_AB=True)

        loss = outputs.loss

        self.log("prompt_train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["enc_inputs"], attention_mask=batch["attn_mask"], \
                            labels=batch["labels"], prompting_AB=True)

        loss = outputs.loss

        loss = outputs.loss
        self.log("val_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx):
        encoder_outputs = self.model(input_ids=batch["enc_inputs"],
                                    attention_mask=batch["attn_mask"],
                                    return_dict=False,
                                    prompting_A=True,
                                    encoder_only=True)
        if isinstance(encoder_outputs, tuple):
            last_hidden_state = encoder_outputs[0]
        else:
            last_hidden_state = encoder_outputs
        
        # Wrap up encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)

        prefix_attention_mask = torch.ones(batch["attn_mask"].size(0), self.cfg.prefix_len).to(self.device)
        batch["attn_mask"] = torch.cat((prefix_attention_mask, batch["attn_mask"]), dim=1)

    
        outputs = self.model.generate(encoder_outputs=encoder_outputs,
                                    attention_mask=batch["attn_mask"],
                                    eos_token_id=self.eos_token_id,
                                    max_length=256,
                                    early_stopping=True,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p)

        outputs = outputs.cpu().numpy().tolist()
        explanations = batch["enc_inputs"].cpu().numpy().tolist()
        gt_qas = batch["labels"].cpu().numpy().tolist()
        decodeds = []
        for n, out in enumerate(outputs):
            exp = explanations[n]
            gt_qa = gt_qas[n]
            out = out[1:out.index(self.eos_token_id)]
            exp = exp[:exp.index(self.eos_token_id)]
            gt_qa = gt_qa[:gt_qa.index(self.eos_token_id)]
            decoded_sample = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
            decoded_exp = self.tokenizer.decode(exp, clean_up_tokenization_spaces=True)
            decoded_gt_qa = self.tokenizer.decode(gt_qa, clean_up_tokenization_spaces=True)
            
            decodeds.append({"input":decoded_exp, "output":decoded_sample, "GT":decoded_gt_qa})
        
        return decodeds


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if self.cfg.optimizer=="adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.cfg.learning_rate, \
                                relative_step=False, scale_parameter=False, warmup_init=False)
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.cfg.learning_rate, eps=self.cfg.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, self.total_steps)

        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]
