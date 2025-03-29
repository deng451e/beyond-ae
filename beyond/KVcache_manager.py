import logging
import threading
import torch 
 
 
def distill(k,v,A,keep_num):
       
        ids = A.topk(keep_num,dim=1).indices  
        # ids, _ = ids.sort()
        dim0_ids = torch.arange(k.size(0))[:, None]
        dim0_ids = dim0_ids.expand_as(ids)
        k = k[dim0_ids, ids]
        v = v[dim0_ids, ids]
        return k,v 
   

class KVCacheManager:
    def __init__(
        self,
        config,  
        device_map,
        batch_size,
        seq_dim=0,
        max_blk_gpu=16,
        max_blk2gpu=2,
        block_size=64,
        head_split_num=4):

        
        self.model = config.model_type
        self.head_num = config.num_attention_heads
        self.head_dim = config.hidden_size//self.head_num  
        self.layer_num = config.num_hidden_layers
        self.dim0 = self.head_num//head_split_num
        
        

        self.seq_dim = seq_dim
        self.batch_size = batch_size 
        self.blk_size = block_size
         
        self.max_blk_gpu = max_blk_gpu # per requests 
        self.max_blk2gpu = max_blk2gpu # per requests 
         
        self.seq_max = block_size*max_blk_gpu
        self.seq2gpu = block_size*max_blk2gpu
        
        self.keep_min = 10
        self.beta  = 0.5 # thresh factor
        self.alpha = 0.5 # merge factor
        self.tasks = []  
        

        # data structure for cpu cache
        self.req_cache = [([],[]) for _ in range(self.layer_num)]  # bh,s,d
        self.cache2gpu = [(None,None) for _ in range(self.layer_num)] # s,bh,d
        self.cache_cpu = [(None,None) for _ in range(self.layer_num)] # s,bh,d
        self.all_ptrs = [(0,0,0) for _ in range(self.layer_num)] # 2gpu_offst, ed_ptr, pos ptr

        # trace per device     
        self.copy_streams = []
        self.cache_gpu = [] # s,bh,d
        self.cache_map = {}
         
        self.init_workspace(device_map)
     

    def init_workspace(self,device_map):
      
        if self.seq_max==0:
            return 
        
        layer_ofst = 0 
        dev_ofst = 0
        device = device_map["0"] 
         
        for layer_idx in range(self.layer_num):
            if device_map[f"{layer_idx}"]!=device:
                k = torch.zeros(layer_ofst,self.seq_max,self.batch_size*self.head_num,self.head_dim,device=device,dtype=torch.float16)
                v = torch.zeros(layer_ofst,self.seq_max,self.batch_size*self.head_num,self.head_dim,device=device,dtype=torch.float16)
                A = torch.zeros(layer_ofst,self.batch_size*self.head_num,self.seq_max+self.seq2gpu,device=device,dtype=torch.float16)
                self.cache_gpu.append((k,v,A))
                self.copy_streams.append(torch.cuda.Stream(device=device))
                device = device_map[f"{layer_idx}"] 
                layer_ofst = 0
                dev_ofst += 1 

            self.cache_map[layer_idx] = (dev_ofst,layer_ofst)
            layer_ofst += 1 

        k = torch.zeros(layer_ofst,self.seq_max,self.batch_size*self.head_num,self.head_dim,device=device,dtype=torch.float16)
        v = torch.zeros(layer_ofst,self.seq_max,self.batch_size*self.head_num,self.head_dim,device=device,dtype=torch.float16)
        A = torch.zeros(layer_ofst,self.batch_size*self.head_num,self.seq_max+self.seq2gpu,device=device,dtype=torch.float16)
        self.cache_gpu.append((k,v,A))
        self.copy_streams.append(torch.cuda.Stream(device=device))
        return  



    def set_beta(self,x):
        self.beta = x 
        return 
    
    def set_alpha(self,x):
        self.alpha = x 
        return 

    def clear_req_cache(self,):
        self.req_cache = [([],[]) for _ in range(self.layer_num)] 
        return
     
    def clear_all_cache(self,):
        self.clear_req_cache()    
        self.cache_cpu = [(None,None) for _ in range(self.layer_num)] 
        self.all_ptrs = [(0,0,0) for _ in range(self.layer_num)]
        return 
    
    def preload(self,layer_idx):
        if self.seq2gpu==0: return 
 
        
        next_layer_idx = (layer_idx+1)%(self.layer_num )
        dev_ofst,_ = self.cache_map[next_layer_idx]
        k_cpu,v_cpu = self.cache_cpu[next_layer_idx]

        if k_cpu is None: return 
        ptr = min(self.seq2gpu,k_cpu.size(1))
        if ptr==0: return 
        with torch.cuda.stream(self.copy_streams[dev_ofst]):
            k2gpu = k_cpu[-ptr:,:,:].to(f"cuda:{dev_ofst}", non_blocking=True)
            v2gpu = v_cpu[-ptr:,:,:].to(f"cuda:{dev_ofst}", non_blocking=True)
            self.cache2gpu[next_layer_idx] = (k2gpu,v2gpu)

        return  
     
      
    def __call__(self,q, layer_idx):
        qs = q.size(-2)
        device = q.device
         
        dev_ofst,layer_ofst = self.cache_map[layer_idx] 
        k_dev,v_dev,_ = self.cache_gpu[dev_ofst]
        k_cache,v_cache  = k_dev[layer_ofst],v_dev[layer_ofst] 
        k_gpu,v_gpu = None,None
        k_cpu,v_cpu = [],[]
        
        ptr_2gpu,ptr_seq,_ = self.all_ptrs[layer_idx]  # 2gpu offst, seq ptr, pos ptr
        
        if ptr_seq:
            k_gpu = k_cache[:ptr_seq,:,:] 
            v_gpu = v_cache[:ptr_seq,:,:] 
            
          
        if ptr_2gpu!=0:  
            # k2gpu,v2gpu = self.cache2gpu[layer_idx]
            k_cpu,v_cpu = self.cache_cpu[layer_idx]
            k2gpu,v2gpu = k_cpu[-ptr_2gpu:,:,:],v_cpu[-ptr_2gpu:,:,:]
            
            if k_gpu is not None:
                k_gpu = torch.concat([k2gpu.to(device, non_blocking=True),k_gpu],dim=0) 
                v_gpu = torch.concat([v2gpu.to(device, non_blocking=True),v_gpu],dim=0) 
            else:
                k_gpu = k2gpu.to(device, non_blocking=True)
                v_gpu = v2gpu.to(device, non_blocking=True)




        if qs==1:
            k_cpu,v_cpu = self.req_cache[layer_idx]
        else:
             
            k_cpu,v_cpu = self.cache_cpu[layer_idx]
            if k_cpu is not None:
                if ptr_2gpu!=0:
                    k_cpu = k_cpu[:-ptr_2gpu,:,:]
                    v_cpu = v_cpu[:-ptr_2gpu,:,:]
                k_cpu = k_cpu.permute(1,0,2)
                v_cpu = v_cpu.permute(1,0,2)
         
        return k_gpu,v_gpu,k_cpu,v_cpu
 
     

    def get_pos(self,layer_idx):
        return self.all_ptrs[layer_idx][2]

     
 
 
   
    
    def print_cache_info(self,layer_idx):
        
        k,_ = self.cache_cpu[layer_idx]
        log = f"CPU size: {max(0,k.size(0)-self.seq2gpu)}" if k is not None else "CPU size: 0"      
        log += f"| GPU size: {self.all_ptrs[layer_idx][1]}"
        if self.seq2gpu: log += f"| 2GPU size: {self.all_ptrs[layer_idx][0]}"
        log += f"| pos: {self.all_ptrs[layer_idx][2]}"
        print(log)


    

    def sync(self,):
        for task in self.tasks:
           task.join()
        return 
    
   
    
    def load_cache(self,k2add,v2add):
        for layer_idx in range(self.layer_num):
            dev_ofst,layer_ofst = self.cache_map[layer_idx]
            k_dev,v_dev,_ = self.cache_gpu[dev_ofst]
            k_gpu,v_gpu = k_dev[layer_ofst],v_dev[layer_ofst]
            seq_ofst = k2add.size(0)
             
            k_gpu[:seq_ofst,:,:] = k2add 
            v_gpu[:seq_ofst,:,:] = v2add 
            ptr_2gpu, _, ptr_pos = self.all_ptrs[layer_idx]
            self.all_ptrs[layer_idx] =( ptr_2gpu, seq_ofst, seq_ofst  )
        
        return 
    
    #############################################
 


    def add_cpu_cache(self,k2cpu,v2cpu,layer_idx):
        if k2cpu is None: return 
        k_cpu,v_cpu = self.cache_cpu[layer_idx] 
        if k_cpu is not None:
 
            k_cpu = torch.concat([k_cpu,k2cpu],dim=0)
            v_cpu = torch.concat([v_cpu,v2cpu],dim=0)
        else:     
            
            k_cpu,v_cpu = k2cpu,v2cpu

        _,ptr_seq,pos_ptr = self.all_ptrs[layer_idx]
        self.all_ptrs[layer_idx] = (min(k_cpu.size(0),self.seq2gpu),ptr_seq,pos_ptr)
        
        self.cache_cpu[layer_idx] = (k_cpu,v_cpu)
        return  

    def append_update(self,k2cpu,v2cpu,A2ofld,A_cpu,layer_idx):
        self.add_cpu_cache(k2cpu,v2cpu,layer_idx)
        ptr_2gpu,ptr_seq,_ = self.all_ptrs[layer_idx] 
        if self.seq2gpu and ptr_2gpu<=self.seq2gpu:
            return 
        
        k_cache_cpu,v_cache_cpu = self.cache_cpu[layer_idx] 
         
        keepnums_cpu,keepnums2cpu = None,None
         
        if A_cpu is not None: 
           
            _,s,ptr = A_cpu.size()
            A_cpu = A_cpu.sum(dim=-2)/s
            k_cpu,v_cpu = k_cache_cpu[:ptr,:].permute(1,0,2),v_cache_cpu[:ptr,:].permute(1,0,2)
            keepnums_cpu = torch.sum(A_cpu >= (1/ptr)*self.beta,dim=-1) 
         
        if A2ofld is not None: 
            
            ptr = A2ofld.size(1)
            
            keepnums2cpu = torch.sum(A2ofld >= (1/(ptr+ptr_2gpu+ptr_seq)*self.beta),dim=-1) 
            
            if self.seq2gpu==0:
               
                k2ofld = k_cache_cpu[-ptr:,:,:] 
                v2ofld = v_cache_cpu[-ptr:,:,:] 
               
            else:   
                k2ofld = k_cache_cpu[-self.seq2gpu-ptr:-self.seq2gpu,:,:] 
                v2ofld = v_cache_cpu[-self.seq2gpu-ptr:-self.seq2gpu,:,:] 

            k2ofld = k2ofld.permute(1,0,2)
            v2ofld = v2ofld.permute(1,0,2)
       
        k_req,v_req = [],[]
        for st in range(0,self.batch_size*self.head_num,self.dim0):
            k2req,v2req = None,None
            ed = st+self.dim0
            if A_cpu is not None:
                keep_num = max(self.keep_min,torch.max(keepnums_cpu[st:ed]))
                 
                k2req,v2req = distill(k_cpu[st:ed,:,:],v_cpu[st:ed,:,:] ,A_cpu[st:ed,:],keep_num )
            if A2ofld is not None:
                 
                keep_num = max(self.keep_min,torch.max(keepnums2cpu[st:ed]))
                # if keep_num>A2ofld.size(0): print(self.seq2gpu,keep_num,ptr,A2ofld.size(),k_cpu.size(),k2ofld.size(),v2ofld.size())
                k2req_,v2req_ = distill(k2ofld[st:ed,:,:],v2ofld[st:ed,:,:] ,A2ofld[st:ed,:],keep_num)


            if A_cpu is not None and A2ofld is not None:
                k2req = torch.concat([k2req,k2req_],dim=1)
                v2req = torch.concat([v2req,v2req_],dim=1)
            elif A2ofld is not None: 
                k2req,v2req = k2req_,v2req_

            k_req.append(k2req)
            v_req.append(v2req)
            
        self.req_cache[layer_idx] = (k_req,v_req)
        return 


    def decode_update(self,k2cpu,v2cpu,A2ofld,layer_idx):
        self.add_cpu_cache(k2cpu,v2cpu,layer_idx)
        ptr_2gpu,ptr_seq,_ = self.all_ptrs[layer_idx]
         
        if self.seq2gpu and ptr_2gpu<=self.seq2gpu:
            return 
        
        k_cpu,v_cpu = self.cache_cpu[layer_idx] 
  
        if self.seq2gpu==0:
            k2req = k_cpu[-self.blk_size:,:,:] 
            v2req = v_cpu[-self.blk_size:,:,:] 
        else:   
            k2req = k_cpu[-self.seq2gpu-self.blk_size:-self.seq2gpu,:,:] 
            v2req = v_cpu[-self.seq2gpu-self.blk_size:-self.seq2gpu,:,:] 
        k2req = k2req.permute(1,0,2)
        v2req = v2req.permute(1,0,2)

        ptr = A2ofld.size(-1)
        keep_nums = torch.sum(A2ofld >= (1/(ptr+ptr_2gpu+ptr_seq)*self.beta),dim=-1) 
        k_req,v_req = self.req_cache[layer_idx]
        inited = True if len(k_req)!=0 else False 
        st,ed = 0,0
        
        for idx,st in enumerate(range(0,self.batch_size*self.head_num,self.dim0)):
            ed = st+self.dim0
            keep_num = max(self.keep_min,torch.max(keep_nums[st:ed]))
            k2add,v2add = distill(k2req[st:ed],v2req[st:ed],A2ofld[st:ed],keep_num)
            if inited:
              
                k_req[idx] = torch.concat([k_req[idx],k2add],dim=1)
                v_req[idx] = torch.concat([v_req[idx],v2add],dim=1)
                
            else:
                k_req.append(k2add)
                v_req.append(v2add)
           
        self.req_cache[layer_idx] = (k_req,v_req)
        return 


    def add_cache(self,layer_idx, k2add, v2add,A_gpu,A_cpu=None):
        # s,bh,d
    
        dev_ofst,layer_ofst = self.cache_map[layer_idx]
        k_cache,v_cache,A = self.cache_gpu[dev_ofst]
        k_cache,v_cache,A = k_cache[layer_ofst],v_cache[layer_ofst],A[layer_ofst]
        ptr_2gpu,ptr_seq,pos_ptr = self.all_ptrs[layer_idx]
         
        with torch.cuda.stream(self.copy_streams[dev_ofst]):     
            len2add = k2add.size(1)
            if len2add==1:
                A_gpu = A_gpu.squeeze(-2)
            else: 
                s = A_gpu.size(-2)
                A_gpu = A_gpu.sum(dim=-2)/s
                
            k2add, v2add = k2add.permute(1,0,2), v2add.permute(1,0,2)
             
            k2cpu,v2cpu,A2ofld = None,None,None
            len2cpu = 0

            if self.seq_max==0:
                k2cpu = k2add.to("cpu",non_blocking=True)
                v2cpu = v2add.to("cpu",non_blocking=True)
                new_ptr_seq = 0
            
            elif ptr_seq+len2add < self.seq_max:
 
                new_ptr_seq = ptr_seq+len2add
                k_cache[ptr_seq:new_ptr_seq,:,:] = k2add
                v_cache[ptr_seq:new_ptr_seq,:,:] = v2add
            
            else:
                len2cpu = (ptr_seq+len2add-self.seq_max+self.blk_size)//self.blk_size*self.blk_size
              
                if len2cpu<ptr_seq:
                    k2cpu = k_cache[:len2cpu,:,:].to("cpu",non_blocking=True)
                    v2cpu = v_cache[:len2cpu,:,:].to("cpu",non_blocking=True)
                    ptr_st = ptr_seq-len2cpu
                    k_cache[:ptr_st,:,:] = k_cache[len2cpu:ptr_seq,:,:].clone()
                    v_cache[:ptr_st,:,:] = v_cache[len2cpu:ptr_seq,:,:].clone()
                    new_ptr_seq = ptr_st + len2add
                    k_cache[ptr_st:new_ptr_seq,:,:] = k2add
                    v_cache[ptr_st:new_ptr_seq,:,:] = v2add

                else:

                    ptr_st =  len2cpu-ptr_seq 
                    if ptr_seq:
                        k2cpu = k_cache[:ptr_seq,:,:] 
                        v2cpu = v_cache[:ptr_seq,:,:] 
                        k2cpu = torch.cat([k2cpu,k2add[:ptr_st,:,:]],dim=0).to("cpu",non_blocking=True)
                        v2cpu = torch.cat([v2cpu,v2add[:ptr_st,:,:]],dim=0).to("cpu",non_blocking=True)
                         
                    else:
                        k2cpu = k2add[:ptr_st,:,:].to("cpu",non_blocking=True)
                        v2cpu = v2add[:ptr_st,:,:].to("cpu",non_blocking=True)

                    new_ptr_seq = len2add-ptr_st
                    k_cache[:new_ptr_seq,:,:] = k2add[ptr_st:,:,:]
                    v_cache[:new_ptr_seq,:,:] = v2add[ptr_st:,:,:]
            

             

            # ptr_2gpu,ptr_seq,pos_ptr
            new_pos_ptr = pos_ptr + len2add
        

            ptr = ptr_seq+ptr_2gpu
            
            if len2add==1:  
                A_gpu[:,:ptr] = A_gpu[:,:ptr]*self.alpha + A[:,:ptr]*(1-self.alpha)
            # print(torch.min(A_gpu),torch.mean(A_gpu),torch.max(A_gpu),1/A_gpu.size(1))

            ptr2fld = ptr_2gpu+len2cpu-self.seq2gpu
            
            ptr = A_gpu.size(1)
            if ptr2fld>0:
                new_ptr_2gpu = self.seq2gpu
                A2ofld = A_gpu[: ,:ptr2fld].to("cpu",non_blocking=True)
                A[:,:ptr-ptr2fld] = A_gpu[:,ptr2fld:]
               
            else:
                new_ptr_2gpu = ptr_2gpu+len2cpu
                A[:,:ptr] = A_gpu 
            # print(len2cpu,ptr2fld,ptr)
             
            self.all_ptrs[layer_idx] = (new_ptr_2gpu,new_ptr_seq,new_pos_ptr)
        
       
        if len2cpu==0: 
            
            return 
        
        
        # Asynchronous launch cpu kvcache managing task 

        if len2add==1: # decode 
            if k2cpu is not None:
                task = threading.Thread(target= self.decode_update, args=(k2cpu,v2cpu,A2ofld,layer_idx))
                task.start()
                self.tasks.append(task)
                    
        else: # append       
            
             
            task = threading.Thread(target= self.append_update, args=(k2cpu,v2cpu,A2ofld,A_cpu,layer_idx))
            task.start()
            self.tasks.append(task)
        
        return 
            
#####################
#      For Test     # 
#####################

 

def run_test(args):

    print_args_info(args)
    config = AutoConfig.from_pretrained(args.model_name)
    batch_size = args.batch_size
    head_num = config.num_attention_heads
    head_dim = config.hidden_size//head_num 
    block_size=args.block_size
    max_blk_gpu=args.max_blocks
    add_len = args.add_len
    max_blk2gpu = args.max_blk2gpu
    device_id = 0
    
    layer_num = config.num_hidden_layers
    device_map = {}
    for i in range(layer_num):
        device_map[str(i)] = "cuda:0"
    
    # with torch.profiler.profile(
    # activities=[
    #     torch.profiler.ProfilerActivity.CPU,  # Capture CPU traces
    #     torch.profiler.ProfilerActivity.CUDA  # Capture GPU traces
    # ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # Save traces for TensorBoard
    # record_shapes=True,  # Record tensor shapes
    # with_stack=True  # Include stack trace (optional, but increases overhead)
    # ) as profiler:
    KVCache_manager = KVCacheManager(config,device_map,batch_size,
                                    max_blk2gpu=max_blk2gpu,
                                    block_size=block_size,
                                    max_blk_gpu=max_blk_gpu)
    KVCache_manager.set_alpha(0.9)
    KVCache_manager.set_beta(1)
    layer_idx  =0 
    add_len = 321#max_blk_gpu*block_size
    k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    A_gpu = torch.softmax(torch.randn(batch_size*head_num,add_len,add_len,dtype=torch.float16).to(f"cuda:{device_id}"),dim=-1)
    KVCache_manager.add_cache(layer_idx,k_add,v_add,A_gpu)
    KVCache_manager.sync()
    KVCache_manager.print_cache_info(0)
    
    q = torch.randn(batch_size*head_num,1,1,dtype=torch.float16).to(f"cuda:{device_id}")
    k_gpu,_,k_cpu,_ = KVCache_manager(q,0)
    for x in k_cpu:
        print(x.shape)
    print("=========================================")
    
    for _ in range(100):
        add_len = 1
        k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
        v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
        A_gpu = torch.softmax(torch.randn(batch_size*head_num,1,k_gpu.size(0)+1,dtype=torch.float16).to(f"cuda:{device_id}"),dim=-1)

        KVCache_manager.add_cache(layer_idx,k_add,v_add,A_gpu)
        # KVCache_manager.preload(i)
        KVCache_manager.sync()
        KVCache_manager.print_cache_info(0)
        
        print("=========================================")
        k_gpu,_,k_cpu,_ = KVCache_manager(k_add,0)
    
        
 
     
     

    # add_len = 128
    # k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    # v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    # A_gpu = torch.softmax(torch.randn(batch_size*head_num,add_len,k_gpu.size(0)+add_len,dtype=torch.float16).to(f"cuda:{device_id}"),dim=-1)


    # KVCache_manager.add_cache(layer_idx,k_add,v_add,A_gpu )
    # # KVCache_manager.preload(i)
    # KVCache_manager.sync()
    # KVCache_manager.print_cache_info(0)
     
    # print("=========================================")
    # k_gpu,_,k_cpu,_ = KVCache_manager(k_add,0)
    

    # add_len = 10
    # k_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    # v_add = torch.randn(batch_size*head_num,add_len,head_dim,dtype=torch.float16).to(f"cuda:{device_id}")
    # A_gpu =torch.softmax(torch.randn(batch_size*head_num,add_len,k_gpu.size(0)+add_len,dtype=torch.float16).to(f"cuda:{device_id}"),dim=-1)
    # A_cpu = torch.softmax(torch.randn(batch_size*head_num,add_len,k_cpu.size(0)+add_len,dtype=torch.float16),dim=-1)
    # KVCache_manager.add_cache(layer_idx,k_add,v_add,A_gpu,A_cpu )
    # # KVCache_manager.preload(i)
    # KVCache_manager.sync()
    # KVCache_manager.print_cache_info(0)
     

    
    
    

    return 
        
    

if __name__ == "__main__":
    import time 
    import logging 
    import argparse        
    import numpy as np
    from beyond.utils import *
    import torch.profiler
    from transformers import AutoConfig
    # torch.cuda.current_device()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
   
    parser.add_argument("--qs", type=int, default=1)

    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--max_blocks", type=int, default=5)
    parser.add_argument("--max_blk2gpu", type=int, default=0)


    parser.add_argument("--add_len", type=int, default=100)
     
     
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default= "facebook/opt-66b")
 
    args = parser.parse_args()

    run_test(args)
     
