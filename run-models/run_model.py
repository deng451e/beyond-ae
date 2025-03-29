import warnings

warnings.filterwarnings("ignore")
import logging
import torch
import argparse
import json
import os
import time
import numpy as np 
from tqdm import tqdm
from beyond.utils import *
from beyond.loading import * 
from beyond.KVcache_manager import KVCache_manager_
 

logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values,KVCache_manager, max_gen_len,enable_beyond):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values if not  enable_beyond else None 
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if enable_beyond:
            past_key_values = None 
            # KVCache_manager.copy_stream.synchronize()

        else:
            past_key_values = outputs.past_key_values
             
         
             
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
          
            break
     
    print(" ".join(generated_text[pos:now]), flush=True)
    return past_key_values,len(generated_text) 


@torch.no_grad()
def inference(model, tokenizer, prompts, KVCache_manager=None, max_gen_len=1000,enable_beyond=False):
    past_key_values = None
    throughputs = []
    memory = []
    for idx, prompt in enumerate(prompts):
        
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        # print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        logger.info(f"================================")
        logger.info(f"Index {idx}: {input_ids.size(1)}")
        logger.info(f"================================")
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        torch.cuda.reset_peak_memory_stats()  # Reset stats for accurate measurement
        torch.cuda.synchronize()

        st = time.time()
        past_key_values,token_cnt = greedy_generate(
            model, tokenizer, input_ids, past_key_values,KVCache_manager, max_gen_len=max_gen_len,enable_beyond=enable_beyond
        )
        memory.append(torch.cuda.max_memory_allocated(model.device)/ (1024 ** 3))
        throughputs.append(token_cnt/(time.time()-st))

        logger.info(f"idx:{idx},memory:{np.mean(memory):4},throughput:{np.mean(throughputs):4}")
         
def main(args):
    
    

    # load dataset 
    prompts,_ = load_dataset_(args.data_root)
    
    # load model 
    print(f"Loading model from {args.model_name_or_path} ...")
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load_model(model_name_or_path)
     
    # load KV manager 
    KVCache_manager = None
    ##############################
    #          beyond            #
    ##############################
    if args.enable_beyond:
        
        from beyond.KVcache_manager import KVCache_manager_
        
        if "llama" in model.config.model_type:
            from beyond.models.modify_llama import modify_llama_attention as modify_attention
             
        elif "opt" in model.config.model_type:
            from beyond.models.modify_opt import modify_opt_attention as modify_attention
        
        elif "gpt_neox" in model.config.model_type:
            from beyond.models.modify_gptNeox import modify_GPTNeoX_attention as modify_attention

        config = model.config 
        KVCache_manager = KVCache_manager_(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=2,
            v_seq_dim=2,
            head_dim=config.hidden_size//config.num_attention_heads,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            gpu_cache_max=200000,
            cpu_attn_size=2000000,
            gpu_cache_device="cuda",
        )
        if args.config_file_path:
            print(f"Loading KV manager from {args.config_file_path} ...")
            KVCache_manager = set_kv_manager_config(KVCache_manager,args.config_file_path)
        KVCache_manager.print_coverage()
        modify_attention(model,KVCache_manager)
    # start inference 
    inference(
        model,
        tokenizer,
        prompts,
        KVCache_manager=KVCache_manager,
        enable_beyond=args.enable_beyond,
    )


if __name__ == "__main__":
    if os.path.exists("stats.log"): os.remove("stats.log")
    logging.basicConfig(filename='stats.log', level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=10000) 
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--config_file_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data/mt_bench.jsonl")
    args = parser.parse_args()
    args.enable_beyond = True
    main(args)