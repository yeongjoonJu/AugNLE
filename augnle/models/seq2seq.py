import torch
import torch.nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.prefix_encoder import PrefixEncoder

class T5PrefixForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, tokenizer):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        
        self.prefix_seq_len = config.prefix_seq_len
        self.n_layers = config.num_hidden_layers#8
        self.n_heads = config.num_attention_heads#6
        self.n_embeds = config.hidden_size // config.num_attention_heads#1024/6
        self.task_name = self.config.task_name
        for param in self.parameters():
            param.requires_grad = False

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.prefix_encoder = PrefixEncoder(config)
        self.prefix_tokens = torch.arange(self.prefix_seq_len).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)#batch,30
        promt_output = self.prefix_encoder(prefix_tokens)
        bsz, seq_len, emb_dim = promt_output.shape
        #promt_output = promt_output.view(bsz, seq_len, self.n_layers*2, self.n_heads, self.n_embeds)
        promt_output = self.dropout(promt_output)
        #past_key_values = past_key_values.permute([2,0,3,1,4]).split(2)

        return promt_output
    
    def get_situation_idx(self, ids):
        situation_token = self.tokenizer.encode("<situation>")[0]
        situation_idx = []
        for i in ids:
            for num,j in enumerate(i):
                if j == torch.Tensor([situation_token]).to(self.device):
                    situation_idx.append(num)
                    break
        return situation_idx
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_only=None,):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        batch_size = input_ids.shape[0]
        prompt_enc = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_seq_len).to(self.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        ids_embeds = self.shared(input_ids)
        
        # Fill situation embedding
        if self.task_name == "task_A":
            sit_idx = self.get_situation_idx(input_ids)
            for idx,location in enumerate(sit_idx):
                ids_embeds[idx][location:location+196, :] = inputs_embeds[idx]
    
        ids_embeds = torch.cat((prompt_enc,ids_embeds),dim = 1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=None,
                                           attention_mask=attention_mask,
                                           inputs_embeds=ids_embeds,
                                           past_key_values=None,
                                           return_dict=return_dict)
            if return_dict:
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs[0]
        else:
            if isinstance(encoder_outputs, tuple):
                encoder_hidden_states = encoder_outputs[0]
            else:
                encoder_hidden_states = encoder_outputs.last_hidden_state
        
        if encoder_only:
            return encoder_outputs
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training"
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            encoder_hidden_states = encoder_hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                  inputs_embeds=decoder_inputs_embeds,
                                  past_key_values=past_key_values,
                                  attention_mask=decoder_attention_mask,
                                  encoder_hidden_states=encoder_hidden_states[:,self.prefix_seq_len:],
                                  encoder_attention_mask=attention_mask[:,self.prefix_seq_len:],
                                  use_cache=use_cache,
                                  return_dict=return_dict)

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        # for training
        if not return_dict:
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (loss, pred_lm, encoder_hidden_states)
        # for prediction
        else:
            outputs = Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                encoder_attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)

        return outputs