from sys import prefix
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
        
        self.prefix_len = config.prefix_len
        self.n_layers = config.num_hidden_layers  # 8
        self.n_heads = config.num_attention_heads # 6
        self.n_embeds = config.hidden_size // config.num_attention_heads#1024/6
        for param in self.parameters():
            param.requires_grad = False

        self.dropout = torch.nn.Dropout(config.prefix_dropout)
        self.prefix_encoder = PrefixEncoder(config)
        self.prefix_seqs = [
            torch.arange(self.prefix_len).unsqueeze(0).long(),
            torch.arange(self.prefix_len, self.prefix_len*2).unsqueeze(0).long()
        ]

    def prepare_inputs_for_generation(self, input_ids,
                                      past=None, attention_mask=None,
                                      use_cache=None, encoder_outputs=None,
                                      **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"decoder_input_ids": input_ids,
                "past_key_values": past,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
                "use_cache": use_cache,}
    
    def get_prompt_A(self, batch_size):
        prefix_tokens = self.prefix_seqs[0].expand(batch_size,-1).to(self.device)
        return self.dropout(self.prefix_encoder(prefix_tokens))
    
    def get_prompt_B(self, batch_size):
        prefix_tokens = self.prefix_seqs[1].expand(batch_size,-1).to(self.device)
        return self.dropout(self.prefix_encoder(prefix_tokens))


    def get_mixed_prompt(self, batch_size):
        prefix_tokens1 = self.prefix_seqs[0].expand(batch_size//2,-1).to(self.device)
        prefix_tokens2 = self.prefix_seqs[1].expand(batch_size-batch_size//2,-1).to(self.device)
        prefix_tokens = torch.cat((prefix_tokens1, prefix_tokens2), dim=0)
        prompt_output = self.prefix_encoder(prefix_tokens)
        prompt_output = self.dropout(prompt_output)

        return prompt_output

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        
        for param in self.prefix_encoder.parameters():
            param.requires_grad = False
    
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
                encoder_only=None,
                prompting_A=False,
                prompting_B=False,
                prompting_AB=False):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        if encoder_outputs is None:
            batch_size = input_ids.shape[0]
            if prompting_AB:
                prefix_embeds = self.get_mixed_prompt(batch_size=batch_size)
            elif prompting_A:
                prefix_embeds = self.get_prompt_A(batch_size=batch_size)
            elif prompting_B:
                prefix_embeds = self.get_prompt_B(batch_size=batch_size)
            
            if prompting_AB or prompting_A or prompting_B:
                prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(self.device)
                inputs_embeds = self.shared(input_ids)
                inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                input_ids=None
                
            encoder_outputs = self.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
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
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=attention_mask,
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
        
        if not return_dict:
            # for training
            pred_lm = torch.argmax(lm_logits, dim=-1)
            outputs = (loss, pred_lm, encoder_hidden_states)
        else:
            # for prediction
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