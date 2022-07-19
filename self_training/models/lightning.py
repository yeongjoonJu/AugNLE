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
        self.mode = None
        self.img_encoded = hparams.img_encoded

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
        elif stage=="test" or stage=="predict":
            self.results_full = []
            self.results_exp = []

            SEG_TOKENS = ['<question>', '<answer>', '<explanation>']
            self.seg_token_ids = self.tokenizer.convert_tokens_to_ids(SEG_TOKENS)
            self.because_token_id = self.tokenizer.convert_tokens_to_ids('Ġbecause')
            self.eos_token_id = [self.tokenizer.eos_token_id]
            self.special_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] + self.seg_token_ids
        
    
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
        scheduler = {"scheduler": scheduler, "interval": "step", "name": "lr"}
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self,  batch, batch_idx):
        if self.img_encoded:
            img_emb = batch["img_embeddings"]
        else:
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
        self.log(f"{self.mode}_train_loss", loss)
            
        return loss

    def validation_step(self, batch, batch_idx):
        if self.img_encoded:
            img_emb = batch["img_embeddings"]
        else:
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
        self.log(f"{self.mode}_val_loss", loss)

        return loss

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
            if prev.item() in self.special_token_ids:
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
    
    def predict_step(self, batch, batch_idx):
        device = torch.cuda.current_device()

        input_ids = batch["input_ids"].to(device)
        segment_ids = batch["segment_ids"].to(device)

        if self.img_encoded:
            img_emb = batch["img_embeddings"].to(device)
        else:
            img = batch["image"].to(device)
            img_emb = self.image_encoder(img)
        
        id = batch["id"]

        do_sample = False

        batch_size = input_ids.shape[0]
        eos_token_id = torch.LongTensor(self.eos_token_id).to(device)
        eos_token_id = eos_token_id.unsqueeze(0).expand(batch_size, -1)
        eos_appear = torch.BoolTensor([False]*batch_size).to(device)
        always_exp = torch.BoolTensor([False]*batch_size).to(device)

        because_token = self.tokenizer.convert_tokens_to_ids('Ġbecause') # integer value
        because_token = torch.LongTensor([because_token]).to(device)
        because_token = because_token.unsqueeze(0).expand(batch_size, -1)

        # For batch prediction
        sample_lens = (input_ids >= 0).sum(dim=1)
        min_sample_len = sample_lens.min()
        max_sample_len = sample_lens.max()

        generated = [[] for _ in range(batch_size)]
        with torch.no_grad():
            for step in range(100 + 1):
                if step == 100:
                    break

                slicing = min_sample_len < max_sample_len
                
                outputs = self.model(input_ids=input_ids[:,:min_sample_len] if slicing else input_ids,
                                    past_key_values=None, 
                                    attention_mask=None, 
                                    token_type_ids=segment_ids[:, :min_sample_len] if slicing else segment_ids,
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

                new_segment = torch.LongTensor([self.seg_token_ids[2]]).to(device)
                new_segment = new_segment.unsqueeze(0).expand(batch_size, -1)

                if not always_exp.prod():
                    always_exp = torch.logical_or(always_exp, torch.eq(prev, because_token).squeeze(-1))
                    new_segment[torch.logical_not(always_exp).unsqueeze(1)] = self.seg_token_ids[1]

                if min_sample_len < max_sample_len:
                    replaced = input_ids[:,min_sample_len]<0
                    prev = prev.squeeze(-1)
                    input_ids[:,min_sample_len][replaced] = prev[replaced]
                    segment_ids[:,min_sample_len][replaced] = new_segment.squeeze(-1)[replaced]
                    min_sample_len += 1
                    for k, r in enumerate(replaced):
                        if r.item():
                            generated[k].append(prev[k].item())
                else:
                    input_ids = torch.cat((input_ids, prev), dim = 1)
                    segment_ids = torch.cat((segment_ids, new_segment), dim = 1)
                    for k, t in enumerate(prev.squeeze(-1)):
                        generated[k].append(t.item())
        
        for k, sample in enumerate(generated):
            generated[k] = {"idx":id[k], "sample": sample}

        return generated