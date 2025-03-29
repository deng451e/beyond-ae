import logging
import types
import torch
from torch import nn
from typing import Optional, Tuple, Union, List

   

from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoder,
    
)
from transformers.modeling_outputs  import (
    BaseModelOutputWithPast,
)


logger = logging.getLogger(__name__)
__all__ = ["modify_llama_attention"]

def modified_OPTDecoder_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape


    
        # Hack the past_key_values_length for masking and pos_emb
        past_key_values_length = self.KVCacheManager.get_pos(0)
        
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length
 
       
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
         
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
def modified_opt_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
   
   
   

    bs, qs, _ = hidden_states.size()  
    q_states = self.q_proj(hidden_states) * self.scaling
    k_states = self.k_proj(hidden_states)
    v_states = self.v_proj(hidden_states)
    q_states = q_states.view(bs, qs, self.num_heads, self.head_dim).transpose(1, 2).reshape(bs*self.num_heads,qs,-1)
    k_states = k_states.view(bs, qs, self.num_heads, self.head_dim).transpose(1, 2).reshape(bs*self.num_heads,qs,-1)
    v_states = v_states.view(bs, qs, self.num_heads, self.head_dim).transpose(1, 2).reshape(bs*self.num_heads,qs,-1)
   
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = self.KVCacheManager(q_states,self.layer_idx) # b,h,s,d
    # if self.layer_idx==0 and k_cache_gpu is not None: print( k_cache_gpu.shape)
    if k_cache_gpu is not None:
        k_cache_gpu = k_cache_gpu.data.permute(1,0,2)
        v_cache_gpu = v_cache_gpu.data.permute(1,0,2)
    
    if qs!=1:
        mask = attention_mask[0,:,:,-qs :]
    else:
        mask = None
    
    attn_output,A_gpu,A_cpu  =  self.hybrid_attn(q_states,k_states,v_states,k_cache_gpu,v_cache_gpu, k_cache_cpu,v_cache_cpu,mask) 
    if not output_attentions:
        attn_weights = None
     
    
    attn_output = attn_output.reshape(bs, qs, self.embed_dim)
    attn_output = self.out_proj(attn_output)
    self.KVCacheManager.add_cache(self.layer_idx, k_states,v_states,A_gpu,A_cpu)
    if self.layer_idx==self.layer_num: self.KVCacheManager.sync()
    return attn_output,attn_weights,past_key_value

  
 
def modify_opt_attention(model,KVCacheManager,hybrid_attn):
    global layer_idx
    config = model.config 
    layer_idx = config.num_hidden_layers-1
    
      
    def replace_layer(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                replace_layer( module,)

            if isinstance(module, OPTDecoder):
                model._modules[name].forward = types.MethodType( modified_OPTDecoder_forward, model._modules[name])
                model._modules[name].KVCacheManager = KVCacheManager

            if isinstance(module, OPTAttention):
                global layer_idx
                model._modules[name].layer_num  = config.num_hidden_layers-1
                model._modules[name].layer_idx  = layer_idx
                model._modules[name].KVCacheManager = KVCacheManager
                model._modules[name].hybrid_attn = hybrid_attn
                model._modules[name].forward = types.MethodType( modified_opt_attention_forward, model._modules[name])
                layer_idx -= 1  
    replace_layer(model)
 
 