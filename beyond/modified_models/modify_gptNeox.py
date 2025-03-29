import math
import logging
import types
import torch
from typing import Optional, Tuple

  
 
logger = logging.getLogger(__name__)


from transformers.models.gpt_neox.modeling_gpt_neox import (
    apply_rotary_pos_emb,
    rotate_half,
    GPTNeoXAttention,
    GPTNeoXRotaryEmbedding,
)
import types

__all__ = ["enable_gpt_neox_pos_shift_attention"]

 


def modified_GPTNeoX_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
):
    
     
    
    # Compute QKV
    # Attention heads [batch, seq_len, hidden_size] --> [batch, seq_len, (np * 3 * head_size)]
    qkv = self.query_key_value(hidden_states)

    # [batch, seq_len, (num_heads * 3 * head_size)] --> [batch, seq_len, num_heads, 3 * head_size]
    new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
    qkv = qkv.view(*new_qkv_shape)

    # b h s d 
    q = qkv[..., : self.head_size].permute(0, 2, 1, 3)
    k = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    v = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)
   
    bs,h,qs,_  = q.size( )  

    q_rot  = q[..., : self.rotary_ndims]
    q_pass = q[..., self.rotary_ndims :]
    k_rot  = k[..., : self.rotary_ndims]
    k_pass = k[..., self.rotary_ndims :]

    # Compute token offset for rotary embeddings (when decoding)
    
    seq_len = qs + self.KVCacheManager.get_pos(self.layer_idx)
    position_ids = position_ids + self.KVCacheManager.get_pos(self.layer_idx)
    
    cos, sin = self.rotary_emb(v, seq_len=seq_len)
    cos,sin = cos.to(q.dtype),sin.to(q.dtype)
    q, k = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)
  
    q = torch.cat((q, q_pass), dim=-1).reshape(bs*h,qs,-1)*self.norm_factor
    k = torch.cat((k, k_pass), dim=-1).reshape(bs*h,qs,-1)
    v = v.reshape(bs*h,qs,-1)
     
    if  qs!=1:
        # print(attention_mask)
        if seq_len > self.bias.shape[-1]:
            self._init_bias(seq_len, device=k.device)
        mask  = self.bias[0, :, seq_len - qs : seq_len, seq_len - qs :seq_len]
      
    else:
        mask = None
    
    k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu = self.KVCacheManager(q,self.layer_idx) # b,h,s,d
    torch.cuda.synchronize(device=q.device) 
    if k_cache_gpu is not None:
        k_cache_gpu = k_cache_gpu.data.permute(1,0,2)
        v_cache_gpu = v_cache_gpu.data.permute(1,0,2)

    
    attn_output,A_gpu,A_cpu  =  self.hybrid_attn(q,k,v,k_cache_gpu,v_cache_gpu, k_cache_cpu,v_cache_cpu,mask) 
    
   
    # Reshape outputs
    attn_output =  attn_output.reshape(bs, qs, self.hidden_size)   
    attn_output = self.dense(attn_output)

    outputs = (attn_output, None)
    if output_attentions:
        outputs += (None,)
    self.KVCacheManager.add_cache(self.layer_idx, k,v,A_gpu,A_cpu)
    if self.layer_idx==self.layer_num: self.KVCacheManager.sync()
     

    return outputs

 

def modify_GPTNeoX_attention(model,KVCacheManager,hybrid_attn):
    global layer_idx
    config = model.config 
    layer_idx = config.num_hidden_layers-1
    def replace_layer(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                replace_layer( module,)
 
            if isinstance(module, GPTNeoXAttention):
                global layer_idx
                
                model._modules[name].layer_num  = config.num_hidden_layers-1
                model._modules[name].layer_idx  = layer_idx
                model._modules[name].KVCacheManager = KVCacheManager
                model._modules[name].hybrid_attn = hybrid_attn
                model._modules[name].forward = types.MethodType( modified_GPTNeoX_attention_forward, model._modules[name])
                layer_idx -= 1  # layer are reverseved travesed 
    replace_layer(model)
 