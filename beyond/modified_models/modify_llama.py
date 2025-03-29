import logging
import types
import math
import torch
from typing import Optional, Tuple 

logger = logging.getLogger(__name__)


from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaAttention,
    rotate_half,
    repeat_kv,
)
 

__all__ = ["modify_llama_attention"]

 
 


def modified_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
  
    bs, qs, _ = hidden_states.size()  
      
    q_states = self.q_proj(hidden_states) 
    k_states = self.k_proj(hidden_states)
    v_states = self.v_proj(hidden_states)
   
    # b,h,s,d
    q_states = q_states.view(bs, qs, self.num_heads, self.head_dim).transpose(1, 2) 
    k_states = k_states.view(bs, qs, self.num_key_value_heads, self.head_dim).transpose(1, 2) 
    v_states = v_states.view(bs, qs, self.num_key_value_heads, self.head_dim).transpose(1, 2).reshape(bs*self.num_key_value_heads,qs,-1)
   
     

     
    kv_seq_len = k_states.shape[-2] + self.KVCacheManager.get_pos(self.layer_idx)
    position_ids = position_ids + self.KVCacheManager.get_pos(self.layer_idx)
    cos, sin = self.rotary_emb(v_states, seq_len=kv_seq_len)
    
    # if self.layer_idx==0: print(position_ids,k_states.shape,v_states.shape)
    q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin, position_ids)
    q_states = q_states.reshape(bs*self.num_heads,qs,-1)/ math.sqrt(self.head_dim)
    k_states = k_states.reshape(bs*self.num_key_value_heads,qs,-1)
    if qs!=1:
         
        mask = attention_mask[0,:,:,-qs :]
        # if self.layer_idx==0: print(attention_mask)
    else:
        mask = None
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = self.KVCacheManager(q_states,self.layer_idx) # b,h,s,d
    torch.cuda.synchronize(device=q_states.device) 
    if k_cache_gpu is not None:
        k_cache_gpu = k_cache_gpu.data.permute(1,0,2)
        v_cache_gpu = v_cache_gpu.data.permute(1,0,2)
    attn_output,A_gpu,A_cpu  =  self.hybrid_attn(q_states,k_states,v_states,k_cache_gpu,v_cache_gpu, k_cache_cpu,v_cache_cpu,mask) 
    
    
    if not output_attentions:
        attn_weights = None
    
    
    attn_output = attn_output.reshape(bs, qs, self.hidden_size) 
     
    attn_output = self.o_proj(attn_output)
    
    if not output_attentions:
        attn_weights = None

    
     
    self.KVCacheManager.add_cache(self.layer_idx, k_states,v_states,A_gpu,A_cpu)
    if self.layer_idx==self.layer_num: self.KVCacheManager.sync()
    return attn_output,attn_weights,past_key_value

  
 
def modify_llama_attention(model,KVCacheManager,hybrid_attn):
    global layer_idx
    config = model.config 
    layer_idx = config.num_hidden_layers-1
 
 
 
    def replace_layer(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                replace_layer( module,)
 
            if isinstance(module, LlamaAttention):
                global layer_idx
               
                model._modules[name].layer_num  = config.num_hidden_layers-1
                model._modules[name].layer_idx  = layer_idx
                model._modules[name].KVCacheManager = KVCacheManager
                model._modules[name].hybrid_attn = hybrid_attn
                model._modules[name].forward = types.MethodType( modified_llama_attention_forward, model._modules[name])
                layer_idx -= 1  
                 
    replace_layer(model)
 
 