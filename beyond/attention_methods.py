 
 
import torch 
import torch.nn as nn
from torch.jit import fork, wait
import flashinfer 

from typing import List,Tuple
 
import os
 
  

def masking_(model_type):
        
    match model_type:
 
        case "llama":
            def fn(A,A_mask):
                qs = A.size(1)
                A[:,:,-qs:] = A[:,:,-qs:] + A_mask  
                return  A

        case "opt":
            def fn(A,A_mask):
                qs = A_mask.size(1)
                A[:,:,-qs:] = A[:,:,-qs:] + A_mask 
                A = torch.max(A, torch.tensor(-65504.0, device=A.device))
                # A = A.to(torch.float32)
                return A

        case "gpt-neox":
            def fn(A,A_mask):
                qs = A.size(1)
                # mask_value  = torch.finfo(A.dtype).min
                # mask_value  = torch.tensor(-65504.0, device=A.device)
                A[:,:,-qs:] = torch.where(A_mask, A[:,:,-qs:], -65504.0)
                
                return A
        case "flexgen-opt":
            def fn(A,A_mask):
                qs = A.size(1)
                A  = torch.where(A_mask, A[:,:,-qs:], -1e4)
                return A
            
    return fn
 

masking = masking_(os.getenv("model_name"))
 
 
 
 

 
def mha_lse_cpu( q: torch.Tensor,st:int,ed:int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    

    k = k.permute(0,2,1) 
    A = q[st:ed,:,:]@k  
    
    A_max, _ = torch.max(A,dim=-1,keepdim=True)   
    A  = A - A_max
    
    e = torch.exp(A).to(v.dtype)
    se = e.sum(dim=-1, keepdim=True)
    
    out = (( e / se) @ v) 
    lse = (torch.log(se) + A_max)* torch.tensor(1.4427) 
    
    return out,lse.squeeze(-1).float() 


 
def mha_lse( q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                    enable_mask: bool, mask: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    

    k = k.permute(0,2,1) 
    A = q@k  
    
    if enable_mask:
        A = masking(A,mask)   
    
    A_max, _ = torch.max(A,dim=-1,keepdim=True)  
    A  = A - A_max
    
    e = torch.exp(A).to(v.dtype)
    se = e.sum(dim=-1, keepdim=True)
    A = e / se
    out = (A @ v) 
    lse = (torch.log(se) + A_max)* torch.tensor(1.4427) 
    
    return out,lse.squeeze(-1).float() ,A


class AttnMethods(nn.Module):

    def __init__(self,batch_size,head_num,head_dim ):
        super(AttnMethods,self).__init__()
        
        self.head_num = head_num
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.split_dim  =  8192//self.head_dim
        self.gpu_mha    = torch.jit.script(self.mha)  
        self.hybrid_decode = torch.jit.script(self.hybrid_decode)  
        self.hybrid_append = torch.jit.script(self.hybrid_append)  

      
    
    
    def merge_state(self,out_gpu: torch.Tensor,lse_gpu: torch.Tensor,out_cpu: torch.Tensor,lse_cpu: torch.Tensor)-> torch.Tensor:
        
        bh = out_gpu.size(1)
        
        # merge appaned states 
        if out_gpu.size(0)!=1:
            out = torch.empty_like(out_gpu)
            out_cpu,lse_cpu = out_cpu.to(out_gpu.device),lse_cpu.to(out_gpu.device)
            for idx in range((bh+self.split_dim-1)//self.split_dim):
                st,ed = idx* self.split_dim,min(bh,idx* self.split_dim+ self.split_dim)
                va,sa = out_gpu[:,st:ed,:].contiguous(),lse_gpu[:,st:ed].contiguous()
                vb,sb = out_cpu[:,st:ed,:].contiguous(),lse_cpu[:,st:ed].contiguous()
                out[:,st:ed,:],_ = flashinfer.merge_state(va,sa,vb,sb)
                                                                        
        # merge decode states    
        else:
              
            for idx in range((bh+self.split_dim-1)//self.split_dim):
                st,ed = idx* self.split_dim,min(bh,idx* self.split_dim+ self.split_dim)
                flashinfer.merge_state_in_place(out_gpu[:,st:ed,:],
                                                lse_gpu[:,st:ed],
                                                out_cpu[:,st:ed,:],
                                                lse_cpu[:,st:ed])
            out = out_gpu
        return out 
 
 
   

    @staticmethod
    def hybrid_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor ,
                          k_cache_gpu: torch.Tensor, v_cache_gpu: torch.Tensor,
                          k_cache_cpu: List[torch.Tensor], v_cache_cpu:List[torch.Tensor],
                          enable_mask: bool, mask: torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
            
        bh, qs, head_dim = q.size()
        
        q_cpu = q.to("cpu" , non_blocking=True)
         

        # launch cpu attn 
        tasks = torch.jit.annotate(List[Tuple[int,int,torch.jit.Future[Tuple[torch.Tensor, torch.Tensor]] ]], [])
        st,ed = 0,0
        
        for idx in range(len(k_cache_cpu)):
            ed = st+k_cache_cpu[idx].size(0)
            tasks.append((st,ed,fork( mha_lse_cpu, q_cpu,st,ed,k_cache_cpu[idx],v_cache_cpu[idx])))
            st = ed 
            
        out_cpu = torch.empty(bh,qs,head_dim,dtype=torch.float16, pin_memory=True ) 
        lse_cpu = torch.empty(bh,qs,dtype=torch.float32, pin_memory=True) 
       
        
        # launch gpu attn 
        k_gpu = torch.cat([k_cache_gpu,k],dim=-2)
        v_gpu = torch.cat([v_cache_gpu,v],dim=-2)
        out_gpu,lse_gpu,A_gpu = mha_lse(q ,k_gpu,v_gpu,enable_mask,mask)
        
         
        for st,ed,task in tasks:
            out_cpu[st:ed,:,:],lse_cpu[st:ed,:]= wait(task)
          
        return out_gpu.permute(1,0,2),lse_gpu.permute(1,0) ,out_cpu.permute(1,0,2),lse_cpu.permute(1,0),A_gpu
    

    @staticmethod
    def hybrid_append(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor ,
                            k_cache_gpu: torch.Tensor, v_cache_gpu: torch.Tensor,
                            k_cache_cpu: torch.Tensor, v_cache_cpu: torch.Tensor,
                            enable_mask: bool, mask: torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        
        q_cpu = q.to("cpu" , non_blocking=True)  
        task = fork( mha_lse, q_cpu,k_cache_cpu,v_cache_cpu,False,mask)
        # launch gpu attn 
        k_gpu = torch.cat([k_cache_gpu,k],dim=-2)
        v_gpu = torch.cat([v_cache_gpu,v],dim=-2)
        out_gpu,lse_gpu,A_gpu = mha_lse(q ,k_gpu,v_gpu,enable_mask,mask)
        out_cpu,lse_cpu,A_cpu = wait(task)

        return out_gpu.permute(1,0,2),lse_gpu.permute(1,0) ,out_cpu.permute(1,0,2),lse_cpu.permute(1,0),A_gpu,A_cpu 
    
    @staticmethod
    def mha(q: torch.Tensor,k: torch.Tensor,v: torch.Tensor, enable_mask: bool, mask: torch.Tensor)-> Tuple[torch.Tensor,torch.Tensor]:
 
        k = k.permute(0,2,1) 
        A = q@k 
         
        if enable_mask:
            
            A =  masking(A,mask)   
         
        A_max, _ = A.max(dim=-1, keepdim=True) 
        A  = A - A_max
        e = torch.exp(A) 
        se = e.sum(dim=-1, keepdim=True)
        A  = e / se
        # A = F.softmax(A ,dim=-1 ).half()
       
        out = (A @ v) 
        
        return out,A
     

   

   
    def forward(self, q,k,v,k_cache_gpu,v_cache_gpu,k_cache_cpu,v_cache_cpu,mask=None):
               

        qs =  q.size(-2) 
        A_gpu,A_cpu = None,None
   
        
        if mask is not None and qs!=1:
            enable_mask = True  
        else:
            enable_mask = False  
            mask = torch.tensor(0,device=q.device)
     
        if k_cache_cpu is not None and len(k_cache_cpu)!=0:

            if qs==1:#decode 
              
                out_gpu,lse_gpu,out_cpu,lse_cpu,A_gpu = self.hybrid_decode(q,k,v,k_cache_gpu , v_cache_gpu ,k_cache_cpu , v_cache_cpu, enable_mask,mask )
            else:   #append 
                out_gpu,lse_gpu,out_cpu,lse_cpu,A_gpu,A_cpu = self.hybrid_append(q,k,v,k_cache_gpu , v_cache_gpu ,k_cache_cpu , v_cache_cpu, enable_mask,mask )


            out = self.merge_state(out_gpu,lse_gpu,out_cpu,lse_cpu)
            torch.cuda.synchronize()   
            out = out.reshape(qs,self.batch_size,self.head_num,self.head_dim).permute(1,0,2,3).contiguous()
            
        else:
                
           
            if k_cache_gpu is not None:
                k_gpu = torch.cat([k_cache_gpu,k],dim=-2)  
                v_gpu = torch.cat([v_cache_gpu,v],dim=-2)  
            else:
                k_gpu = k
                v_gpu = v
            out,A_gpu = self.mha(q,k_gpu,v_gpu,enable_mask,mask)
            out = out.reshape(-1,self.head_num,qs,self.head_dim).permute(0,2,1,3).contiguous()
            
       
        return out,A_gpu,A_cpu
 



#####################
#      For Test     # 
#####################
 


def run_test(args):

     
    print(f"number of threads: {torch.get_num_threads()}")
    
    config = AutoConfig.from_pretrained(args.model_name)
     
    print_args_info(args)
    

     
    config = config
    repeat = args.repeat
    batch_size = args.batch_size
    head_num = config.num_attention_heads
    head_dim = config.hidden_size//head_num 
    gpu_cache_size = args.gpu_cache_size
    cpu_cache_size = args.cpu_cache_size
    qs = args.qs
    head_split_num = args.head_split_num
    print(head_num)
    # global apply_pos_emb
    # global masking   

    id = 0
    
    
    hybrid_attn  = AttnMethods( batch_size, head_num,head_dim)
  
    
    
    q = torch.randn(batch_size*head_num,qs,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    k = torch.randn(batch_size*head_num,qs,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    v = torch.randn(batch_size*head_num,qs,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    k_cache_cpu_   = torch.randn(batch_size*head_num,cpu_cache_size,head_dim,dtype=torch.float16)
    v_cache_cpu_   = torch.randn(batch_size*head_num,cpu_cache_size,head_dim,dtype=torch.float16)

    k_cache_gpu = torch.randn(batch_size*head_num,gpu_cache_size,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    v_cache_gpu = torch.randn(batch_size*head_num,gpu_cache_size,head_dim,dtype=torch.float16).to(f"cuda:{id}")
    
    hdim = head_num//head_split_num

    if qs==1:
        k_cache_cpu = [k_cache_cpu_[idx*hdim:idx*hdim+hdim,:,:] for idx in  range(batch_size*head_split_num)]
        v_cache_cpu = [v_cache_cpu_[idx*hdim:idx*hdim+hdim,:,:] for idx in  range(batch_size*head_split_num)]
    else:
        k_cache_cpu = k_cache_cpu_
        v_cache_cpu = v_cache_cpu_
     
    
    

    k_cache_cpu_gpu = k_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim)
    v_cache_cpu_gpu = v_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim)

    ops = calculate_required_flop(batch_size, qs,gpu_cache_size+cpu_cache_size, head_num,head_dim)
 
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q
        k_ = k
        v_ = v
        k_gpu = torch.cat([k_cache_cpu_gpu,k_cache_gpu,k_],dim=-2)
        v_gpu = torch.cat([v_cache_cpu_gpu,v_cache_gpu,v_],dim=-2)
        out_ref,_   = hybrid_attn.mha(q_,k_gpu,v_gpu,False,None )
        out_ref =  out_ref.reshape(-1, head_num,qs, head_dim).permute(0,2,1,3) 
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_gpu
        del v_gpu
    time_taken = np.mean(t[repeat//2:])
    print(f"naive gpu attn time: {time_taken}, flops:{ops/time_taken} ")


    
    
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q
        k_ = k
        v_ = v
        k_gpu = torch.cat([k_cache_cpu_gpu,k_cache_gpu,k_],dim=-2)
        v_gpu = torch.cat([v_cache_cpu_gpu,v_cache_gpu,v_],dim=-2)
        output = torch.nn.functional.scaled_dot_product_attention(q_,k_gpu,v_gpu, is_causal=False)

        
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_gpu
        del v_gpu
        del output
    time_taken = np.mean(t[repeat//2:])
    print(f"torch gpu attn time: {time_taken}, flops:{ops/time_taken} ")



    
    k_cache_gpu_cpu = k_cache_gpu.cpu()
    v_cache_gpu_cpu = v_cache_gpu.cpu()
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q.cpu()
        k_ = k.cpu()
        v_ = v.cpu()
        k_cpu = torch.cat([k_cache_cpu_.view(-1,cpu_cache_size,head_dim),k_cache_gpu_cpu,k_],dim=-2)
        v_cpu = torch.cat([v_cache_cpu_.view(-1,cpu_cache_size,head_dim),v_cache_gpu_cpu,v_],dim=-2)
        out_ref,_   = hybrid_attn.mha(q_,k_cpu,v_cpu,False,None )
        out_ref =  out_ref.reshape(-1, head_num,qs, head_dim).permute(0,2,1,3)
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_cpu
        del v_cpu
        
    time_taken = np.mean(t[repeat//2:])
    print(f"naive cpu attn time: {time_taken}, flops:{ops/time_taken} ")
 
    
    t = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q.cpu()
        k_ = k.cpu()
        v_ = v.cpu()
        k_cpu = torch.cat([k_cache_cpu_.view(-1,cpu_cache_size,head_dim),k_cache_gpu_cpu,k_],dim=-2)
        v_cpu = torch.cat([v_cache_cpu_.view(-1,cpu_cache_size,head_dim),v_cache_gpu_cpu,v_],dim=-2)
        output = torch.nn.functional.scaled_dot_product_attention(q_,k_cpu,v_cpu, is_causal=False)
        
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_cpu
        del v_cpu
    time_taken = np.mean(t[repeat//2:])
    print(f"torch cpu attn time: {time_taken}, flops:{ops/time_taken} ")
   
        
    t = []
    
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        q_ = q
        k_ = k
        v_ = v
        k_gpu = torch.cat([k_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim),k_cache_gpu,k_],dim=-2)
        v_gpu = torch.cat([v_cache_cpu_.to(f"cuda:{id}").view(-1,cpu_cache_size,head_dim),v_cache_gpu,v_],dim=-2)
        out_ref,_   = hybrid_attn.mha(q_,k_gpu,v_gpu,False,None)
        out_ref =  out_ref.reshape(-1, head_num,qs, head_dim).permute(0,2,1,3) 
        torch.cuda.synchronize()
        t.append(time.time()-st)
        del k_gpu
        del v_gpu
        
    
    time_taken = np.mean(t[repeat//2:])
    print(f"load gpu attn time: {time_taken}, flops:{ops/time_taken} ")
     
    # print(len(k_cache_cpu),k_cache_cpu[0].shape,k_cache_cpu[0].is_contiguous())
    # thre = int(cpu_cache_size*1)
    # k_cache_cpu = [x[:,:thre,:].to(f"cuda:{1}").contiguous() for x in k_cache_cpu]
    # v_cache_cpu = [x[:,:thre,:].to(f"cuda:{1}").contiguous() for x in v_cache_cpu]
    # print(len(k_cache_cpu),k_cache_cpu[0].shape,k_cache_cpu[0].is_contiguous())
    # with torch.profiler.profile(
    # activities=[
    #     torch.profiler.ProfilerActivity.CPU,  # Capture CPU traces
    #     torch.profiler.ProfilerActivity.CUDA  # Capture GPU traces
    # ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # Save traces for TensorBoard
    # record_shapes=True,  # Record tensor shapes
    # with_stack=True  # Include stack trace (optional, but increases overhead)
    # ) as profiler:
    # thre = int(cpu_cache_size*0.1)
    # k_cache_cpu = [x[:,:thre-i*10,:].contiguous()  for i,x in enumerate(k_cache_cpu)]
    # v_cache_cpu = [x[:,:thre-i*10,:].contiguous() for i,x in enumerate(v_cache_cpu)]
    
    t = []
    torch.cuda.reset_peak_memory_stats() 
    for _ in range(repeat):
        torch.cuda.synchronize()
        st = time.time()
        out_test,_,_  =  hybrid_attn(q,k,v,k_cache_gpu,v_cache_gpu, k_cache_cpu,v_cache_cpu) 
        torch.cuda.synchronize()
        t.append(time.time()-st)  
     
    memory = (torch.cuda.max_memory_allocated("cuda:0") / 1024**2) 
       
    
    
    time_taken = np.mean(t[repeat//2:])
    print(f"hybrid attn time: {time_taken}, flops:{ops/time_taken} ")
 

    acc = check_eq(out_test,out_ref)  
    assert  (acc>0.9),  f"accuracy {acc*100:.4}%, merge state fail..."
    print(f"Merge accuracy {acc*100:.4}%")


    
    




if __name__ == "__main__": 
    
    import time 
    import logging 
    import argparse        
    import numpy as np
    import torch.profiler
    from beyond.utils import *
    from transformers import AutoConfig
    torch.cuda.current_device()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--qs", type=int, default=1)
    parser.add_argument("--head_split_num", type=int, default=4)
    parser.add_argument("--cpu_cache_size", type=int, default=2048)
    parser.add_argument("--gpu_cache_size", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default= "facebook/opt-66b")
    args = parser.parse_args()

    run_test(args)
     