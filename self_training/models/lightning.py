import json
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import (
    get_linear_schedule_with_warmup,
    GPT2Tokenizer, AutoConfig
)
from modules import ImageEncoder
from gpt import GPT2LMHeadModel
from utils import top_filtering, filter_and_get_scores
import os

class Self_training(LightningModule):
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
        self.learning_rate= self.hparams.learning_rate
        self.adam_epsilon= self.hparams.adam_epsilon
        self.warmup_steps= self.hparams.warmup_steps
        self.weight_decay= self.hparams.weight_decay
        self.train_batch_size= self.hparams.train_batch_size
        self.eval_batch_size= self.hparams.eval_batch_size
        self.img_size = self.hparams.img_size


        #Configuration
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.config = AutoConfig.from_pretrained('distilgpt2')
        self.config.img_size = self.img_size
        self.config.add_cross_attention = True
        self.model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # else:
        #     model_name = 'nle_model_{}'.format(str(self.hparams.epoch))
        #     tokenizer_name = 'nle_gpt2_tokenizer_0'
        #     self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        #     self.model = GPT2LMHeadModel.from_pretrained(self.ckpt_path + self.hparams.model_name)
        self.img_transform = transforms.Compose([transforms.Resize((self.img_size,self.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.image_encoder = ImageEncoder(self.device)
        self.change_requires_grad(self.image_encoder, False)
    
    def change_requires_grad(self, model, req_grad):
        for p in model.parameters():
            p.requires_grad = req_grad
            
    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self,  batch, batch_idx):
        _, input_ids, labels, segment_ids,  img = batch.values()
        image_embedding = self.image_encoder(img)

        outputs = self(input_ids=input_ids,
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=segment_ids, 
                        position_ids=None,
                        encoder_hidden_states=image_embedding, 
                        encoder_attention_mask=None, 
                        labels=labels, 
                        use_cache=False, 
                        return_dict=True)
    
        loss = outputs.loss
            
        # self.log(f"{self.hparams.selfe_mode}_train_loss", loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, input_ids, labels, segment_ids,  img = batch.values()
        image_embedding = self.image_encoder(img)

        outputs = self(input_ids=input_ids,
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=segment_ids, 
                        position_ids=None,
                        encoder_hidden_states=image_embedding, 
                        encoder_attention_mask=None, 
                        labels=labels, 
                        use_cache=False, 
                        return_dict=True)

        
        loss = outputs.loss
        # self.log(f"{self.hparams.selfe_mode}_val_loss", loss)
        self.log("val_loss", loss)

        return loss
    
    def predict_step(self, batch, batch_idx):
        qid, input_ids, labels, segment_ids,  img = batch.values()
        image_embedding = self.image_encoder(img)
        because_token = self.tokenizer.convert_tokens_to_ids('Ġbecause')
        max_len = 20
        always_exp = False
        no_sample = True
        current_output = []
        for step in range(max_len + 1):
            if step == max_len:
                break
            
            outputs = self.model(input_ids=input_ids, 
                            past_key_values=None, 
                            attention_mask=None, 
                            token_type_ids=segment_ids, 
                            position_ids=None, 
                            encoder_hidden_states=image_embedding, 
                            encoder_attention_mask=None, 
                            labels=None, 
                            use_cache=False, 
                            return_dict=True)
            
            lm_logits = outputs.logits 
            logits = lm_logits[0, -1, :] / self.hparams.temperature
            logits = top_filtering(logits, top_k= self.hparams.top_k, top_p= self.hparams.top_p)
            probs = F.softmax(logits, dim=-1)
            prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
            if prev.item() in self.special_tokens_ids:
                break     
    
            if not always_exp:
                
                if prev.item() != because_token:
                    new_segment = self.special_tokens_ids[-2]   # answer segment
                else:
                    new_segment = self.special_tokens_ids[-1]   # explanation segment
                    always_exp = True
            else:
                new_segment = self.special_tokens_ids[-1]   # explanation segment     
                
            new_segment = torch.LongTensor([new_segment]).to(torch.cuda.current_device())
            current_output.append(prev.item())
            input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
            segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)

        input = input_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        qid = qid[0][0]
        decodeds = []   
        for n, out in enumerate([current_output]):
            gt_qa = [label for label in labels[n] if label != -100]
            # out = out[out.index(self.eos_token_id):]
            # inp = input[:input.index(self.pad_token_id)]
            gt_qa = gt_qa[:gt_qa.index(self.eos_token_id)]
            decoded_sample = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
            decoded_inp = self.tokenizer.decode(input[n], clean_up_tokenization_spaces=True)
            decoded_gt_qa = self.tokenizer.decode(gt_qa, clean_up_tokenization_spaces=True)
            
            decodeds.append({"qid": qid, "input":decoded_inp, "output":decoded_sample, "GT":decoded_gt_qa})
        
        return decodeds


    def test_step(self,batch,batch_idx):
        qid, input_ids, labels, segment_ids,  img = batch.values()
        image_embedding = self.image_encoder(img)
        because_token = self.tokenizer.convert_tokens_to_ids('Ġbecause')
        max_len = 20
        always_exp = False
        no_sample = True
        current_output = []
        qid = qid[0][0]
        for step in range(max_len + 1):
            if step == max_len:
                break
            
            outputs = self.model(input_ids=input_ids, 
                            past_key_values=None, 
                            attention_mask=None, 
                            token_type_ids=segment_ids, 
                            position_ids=None, 
                            encoder_hidden_states=image_embedding, 
                            encoder_attention_mask=None, 
                            labels=None, 
                            use_cache=False, 
                            return_dict=True)
            
            lm_logits = outputs.logits 
            logits = lm_logits[0, -1, :] / self.hparams.temperature
            logits = top_filtering(logits, top_k= self.hparams.top_k, top_p= self.hparams.top_p)
            probs = F.softmax(logits, dim=-1)
            prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
            if prev.item() in self.special_tokens_ids:
                break     
    
            if not always_exp:
                
                if prev.item() != because_token:
                    new_segment = self.special_tokens_ids[-2]   # answer segment
                else:
                    new_segment = self.special_tokens_ids[-1]   # explanation segment
                    always_exp = True
            else:
                new_segment = self.special_tokens_ids[-1]   # explanation segment     
                
            new_segment = torch.LongTensor([new_segment]).to(torch.cuda.current_device())
            current_output.append(prev.item())
            input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
            segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
    
        decoded_sequences = self.tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        self.results_full.append({"image_id": qid, "caption": decoded_sequences})
        
        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
        
        self.results_exp.append({"image_id": qid, "caption": cut_decoded_sequences})         
        return {"reults_full" : self.results_full, "results_exp": self.results_exp}

    def test_epoch_end(self, batch_parts):
        resFileExp = os.path.join(self.hparams.output_dir , 'captions_exp_{self.hparams.selfe_mode}'+ '.json')
        unf_resFileExp = os.path.join(self.hparams.output_dir , 'unf_captions_exp_{self.hparams.selfe_mode}' + '.json') 
        unf_resFileFull = os.path.join(self.hparams.output_dir , 'unf_captions_full_{self.hparams.selfe_mode}'  + '.json')
        save_scores_pathExp = os.path.join(self.hparams.output_dir , 'scores_exp_{self.hparams.selfe_mode}' + '.json')
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(self.results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(self.results_full, w)
        
        filter_and_get_scores(resFileExp, save_scores_pathExp, self.results_full, self.results_exp, self.hparams.selfe_mode)    
    
    def setup(self,stage):
        self.results_full = []
        self.results_exp = []
        self.tokenizer = self.trainer.datamodule.tokenizer
        train_loader = self.trainer.datamodule.train_dataloader()
        SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        # Setting
        tb_size = self.hparams.train_batch_size  * self.trainer.accumulate_grad_batches * max(1, self.trainer.gpus)
        self.total_steps = (len(train_loader.dataset) // tb_size) * self.trainer.max_epochs
        self.warmup_steps = int(len(train_loader.dataset) / self.trainer.gpus * self.trainer.max_epochs * 0.2)


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