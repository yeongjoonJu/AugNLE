import json
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import (
    top_k_top_p_filtering,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer, AutoConfig
)
from models.modules import ImageEncoder
from models.gpt import GPT2LMHeadModel
from models.utils import top_filtering, filter_and_get_scores
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import os 
import torchvision.transforms as transforms
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm

class NLX_GPT(LightningModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer

        config = AutoConfig.from_pretrained('distilgpt2')

        # Add configs
        setattr(config, 'img_size', None)
        # setattr(config, 'max_seq_len', None)   
        config.img_size = hparams.img_size
        # config.max_seq_len = hparams.max_seq_len 
        config.add_cross_attention = True

        # Load model
        self.model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.image_encoder = ImageEncoder(self.device)


    def setup(self,stage):
        if stage=="fit":
            self.total_steps = self.hparams.total_steps
            self.warmup_steps = self.hparams.warmup_steps
        elif stage=="test":
            self.results_full = []
            self.results_exp = []

            SEG_TOKENS = ['<question>', '<answer>', '<explanation>']
            self.seg_token_ids = self.tokenizer.convert_tokens_to_ids(SEG_TOKENS)
            self.because_token_id = self.tokenizer.convert_tokens_to_ids('Ġbecause')
            self.eos_token_id = [self.tokenizer.eos_token_id]
        
    
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

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self,  batch, batch_idx):
        # image_embedding = self.image_encoder(batch["image"])
        img_emb = self.image_encoder(batch["image"])
        outputs = self(input_ids=batch["inputs"],
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=batch["segment_ids"], 
                        position_ids=None,
                        encoder_hidden_states=img_emb, 
                        encoder_attention_mask=None, 
                        labels=batch["labels"], 
                        use_cache=False, 
                        return_dict=True)
    
        loss = outputs.loss
            
        # self.log(f"{self.hparams.selfe_mode}_train_loss", loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # image_embedding = self.image_encoder(batch["image"])
        img_emb = self.image_encoder(batch["image"])
        outputs = self(input_ids=batch["inputs"],
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=batch["segment_ids"], 
                        position_ids=None,
                        encoder_hidden_states=img_emb, 
                        encoder_attention_mask=None, 
                        labels=batch["labels"], 
                        use_cache=False, 
                        return_dict=True)
        
        loss = outputs.loss
        # self.log(f"{self.hparams.selfe_mode}_val_loss", loss)
        self.log("val_loss", loss)

        return loss

    def test_step(self,batch,batch_idx):
        img, img_id, input_ids, segment_ids = batch
        image_embedding = self.image_encoder(img)
        
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
            if prev.item() == self.eos_token_id:
                break     
    
            if not always_exp:
                if prev.item() != self.because_token_id:
                    new_segment = self.seg_token_ids[1]   # answer segment
                else:
                    new_segment = self.seg_token_ids[2]   # explanation segment
                    always_exp = True
            else:
                new_segment = self.seg_token_ids[2]   # explanation segment
                
            new_segment = torch.LongTensor([new_segment]).to(torch.cuda.current_device())
            current_output.append(prev.item())
            input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
            segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
    
        decoded_sequences = self.tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        self.results_full.append({"image_id": img_id.item(), "caption": decoded_sequences})
        
        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = " ".join(decoded_sequences.split()[2:])
        
        self.results_exp.append({"image_id": img_id.item(), "caption": cut_decoded_sequences})         
        return {"reults_full" : self.results_full, "results_exp": self.results_exp}

    def test_epoch_end(self, batch_parts):
        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)

        resFileExp = os.path.join(self.hparams.output_dir , 'captions.json')
        unf_resFileExp = os.path.join(self.hparams.output_dir , 'unf_captions.json') 
        unf_resFileFull = os.path.join(self.hparams.output_dir , 'unf_captions_full.json')
        save_scores_pathExp = os.path.join(self.hparams.output_dir , 'scores.json')
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(self.results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(self.results_full, w)
        
        filter_and_get_scores(resFileExp, save_scores_pathExp, self.results_full, self.results_exp, self.hparams.nle_test_anno_path)


class Self_training(LightningModule):
    def __init__(
        self, hparams, **kwargs,):
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

        SEG_TOKENS = ['<question>', '<answer>', '<explanation>']
        self.seg_token_ids = self.tokenizer.convert_tokens_to_ids(SEG_TOKENS)
        
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
        # image_embedding = self.image_encoder(batch["image"])
        img_emb = self.image_embedding(batch["image"])
        outputs = self(input_ids=batch["inputs"],
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=batch["segment_ids"], 
                        position_ids=None,
                        encoder_hidden_states=img_emb, 
                        encoder_attention_mask=None, 
                        labels=batch["labels"], 
                        use_cache=False, 
                        return_dict=True)
    
        loss = outputs.loss
            
        # self.log(f"{self.hparams.selfe_mode}_train_loss", loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # image_embedding = self.image_encoder(batch["image"])
        img_emb = self.image_embedding(batch["image"])
        outputs = self(input_ids=batch["inputs"],
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=batch["segment_ids"], 
                        position_ids=None,
                        encoder_hidden_states=img_emb, 
                        encoder_attention_mask=None, 
                        labels=batch["labels"], 
                        use_cache=False, 
                        return_dict=True)
        
        loss = outputs.loss
        # self.log(f"{self.hparams.selfe_mode}_val_loss", loss)
        self.log("val_loss", loss)

        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch["inputs"]
        ori_input_ids = input_ids.clone().cpu().numpy().tolist()
        segment_ids = batch["segment_ids"]
        
        img_emb = self.image_embedding(batch["image"])
        because_token = self.tokenizer.convert_tokens_to_ids('Ġbecause')
        always_exp = False
        do_sample = False
        
        batch_size = input_ids.shape[0]
        eos_token_id = torch.LongTensor([self.eos_token_id])
        eos_token_id = eos_token_id.expand(batch_size).to(torch.cuda.current_device())
        eos_appear = torch.BoolTensor([False]*batch_size)
        outputs = None
        for step in range(128 + 1):
            if step == 128:
                break
            
            outputs = self.model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=segment_ids, 
                                position_ids=None, 
                                encoder_hidden_states=img_emb, 
                                encoder_attention_mask=None, 
                                labels=None, 
                                use_cache=False, 
                                return_dict=True)
            
            lm_logits = outputs.logits 
            logits = lm_logits[:, -1, :] / self.hparams.temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k= self.hparams.top_k, top_p= self.hparams.top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            prev = torch.multinomial(probs, dim=-1) if do_sample else torch.argmax(probs, dim=-1).unsqueeze(-1)
            eos_appear = torch.logical_or(eos_appear, torch.eq(prev, eos_token_id))
            if eos_appear.prod():
                break
                
            if not always_exp:
                if prev.item() != because_token:
                    new_segment = self.seg_token_ids[1]   # answer segment
                else:
                    new_segment = self.seg_token_ids[2]   # explanation segment
                    always_exp = True
            else:
                new_segment = self.seg_token_ids[2]   # explanation segment
                
            new_segment = torch.LongTensor([new_segment]).to(torch.cuda.current_device())
            new_segment = new_segment.expand(batch_size)

            input_ids = torch.cat((input_ids, prev), dim = 1)
            segment_ids = torch.cat((segment_ids, new_segment), dim = 1)
            if outputs is None:
                outputs = prev
            else:
                outputs = torch.cat((outputs, prev), dim=1)

        # labels = batch["labels"].cpu().numpy().tolist()
        # outputs = outputs.cpu().numpy().tolist()

        # decodeds = []
        # for b, out in enumerate(outputs):
        #     label = [id for id in labels[b] if id != -100]
        #     label = label[:label.index(self.eos_token_id)]
        #     out = out[:out.index(self.eos_token_id)]
        #     input_sample = ori_input_ids[b]
        #     input_sample = input_sample[:input_sample.index(self.eos_token_id)]
        #     decoded_out = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
        #     decoded_in = self.tokenizer.decode(input_sample, clean_up_tokenization_spaces=True)
        #     decoded_label = self.tokenizer.decode(label, clean_up_tokenization_spaces=True)
            
        #     decodeds.append({"image_path": batch["image_path"][b], \
        #         "input":decoded_in, "output":decoded_out, "label":decoded_label})
        
        # return decodeds
        return outputs.cpu()


    def image_embedding(self,qid):
        img_file = qid[0][0].split("/")[0]
        cached_filename = f"{img_file}.cache"
        image_file = os.path.join(self.hparams.cached_dir,cached_filename)
        # caching loading
        if os.path.exists(image_file):
            img_dict = torch.load(image_file)
        else:
            img_dict = self.img_caching(os.path.join(self.hparams.image_dir,img_file))
            
        for idx, img_id in enumerate(qid):    
            if idx == 0:
                img = img_dict[img_id[0]]
            else:
                img = torch.stack(img,img_dict[img_id[0]])
        img.to(torch.cuda.current_device())
        return img
                
    def img_caching(self,img_dir):
        file_lst = os.listdir(img_dir)
        emb_dict = {}
        for file in tqdm(file_lst, desc= f"Processing image..."):
            file_pth = os.path.join(img_dir, file)
            img = Image.open(file_pth).convert('RGB')
            img = self.img_transform(img).unsqueeze(0).to(torch.cuda.current_device())
            img_emb = self.image_encoder(img)
            emb_dict[file]= img_emb
        return emb_dict


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
        # self.tokenizer = self.trainer.datamodule.tokenizer
        # SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
        # self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        # # Setting
        self.total_steps = self.args.total_steps
        self.warmup_steps = self.args.warmup_steps

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