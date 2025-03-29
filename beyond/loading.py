

import torch  
import os.path as osp
import ssl
import urllib.request
import os
import json
  

def set_kv_manager_config( KVCache_manager,file_path=None):
    
    if file_path is  None: return KVCache_manager
    print(f"Loading KV manager from {args.config_file_path} ...")
    with open(file_path, 'r') as file:
        config = json.load(file)
    
    KVCache_manager.gpu_cache_device = config['gpu_cache_device']
    for idx in range(KVCache_manager.num_layers):
        KVCache_manager.start_sizes[idx] = config[f"layer {idx}"]["start_size"]
        KVCache_manager.recent_sizes[idx] = config[f"layer {idx}"]["recent_size"]
        KVCache_manager.cpu_attn_sizes[idx] = config[f"layer {idx}"]["cpu_attn_size"]
        KVCache_manager.blk_num[idx] = config[f"layer {idx}"]["block_size"]

    return KVCache_manager  
 
def load_model(model_name_or_path):
    from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    )

    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_json(file_path):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


 
 


def load_dataset_(dataset_path,cnt=100):
    prompts = []
    outputs = []
    print(f"Loading data from {dataset_path} ...")
     
    match dataset_path:
        case "data/mt_bench.jsonl":
            test_filepath = os.path.join(dataset_path)
             
            if not os.path.exists(test_filepath):
                download_url("https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",'data/',)
                os.rename(os.path.join('data/', "question.jsonl"), test_filepath)

            list_data = load_json(test_filepath)
            for sample in list_data:
                prompts += sample["turns"]
                
        case "hakurei/open-instruct-v1":
            from datasets import load_dataset
            list_data = load_dataset(dataset_path)['train']
            # truncate datset size           
            
            for sample in list_data :
                
                hold =  sample['instruction']+ ': '+ sample['input'] if sample['input'] else sample['instruction']
                prompts += [hold]
                outputs += [sample['output']]
                
                cnt -= 1 
                if cnt==0:
                    break
        case "facebook/content_rephrasing":
            from datasets import load_dataset
            list_data = load_dataset(dataset_path)['train']
            # truncate datset size           
            
            for sample in list_data :
               
              
                prompts += [ sample['Input']]
               
                 
                
                cnt -= 1 
                if cnt==0:
                    break

    return prompts,outputs


 