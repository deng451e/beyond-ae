import re
import pandas as pd
import seaborn as sns 
import numpy as np
import torch
import matplotlib.pyplot as plt
def parse_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split based on the header lines
    sections = re.split(r'={10,}\n', content)
    
    data = []
    
    for section in sections:
        
        lines = section.strip().split('\n')
       
        if not lines :
            continue
        
        # Extract parameters from the first line

        if '--qs'  in lines[0]:
            params = dict(re.findall(r'--(\S+)\s+(\S+)', lines[0]))
            
            continue
        # Extract performance metrics
        for line in lines[1:]:
            match = re.match(r'([\w\s]+):\s+([\d\.]+),\s+flops:([\d\.]+)', line)
            if match:
                attn_type, time, flops = match.groups()
                hold = params |{'attn_type': attn_type.strip(), 'time': float(time), 'flops': float(flops)}
                
                data.append(hold)
   
    return pd.DataFrame(data)
 
fig, axs = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True)
axs = axs.flatten()
 
for model_idx,model in enumerate(["30","66"]):
    file_path=f"/home/wxd/code/Beyond/evaluation/efficiency-test/ablation-test/opt-{model}b.log"
    df = parse_data(file_path)
    gpu_size = sorted([int(x) for x in list(set(df['gpu_cache_size'].values))])
    cpu_size = sorted([int(x) for x in list(set(df['cpu_cache_size'].values))])
     
    for idx,bsz in enumerate([1,5,10]):
        y = []
        mat = np.zeros((len(gpu_size),len(cpu_size)))
        for i,gpu_s in enumerate(gpu_size):
            for j,cpu_s in enumerate(cpu_size):
                t1 = df[df['batch_size']==str(bsz)][df['gpu_cache_size']==str(gpu_s)][df['cpu_cache_size']==str(cpu_s)][df['attn_type']=="load gpu attn time"]['time'] 
                t2 = df[df['batch_size']==str(bsz)][df['gpu_cache_size']==str(gpu_s)][df['cpu_cache_size']==str(cpu_s)][df['attn_type']=="hybrid attn time"]['time'] 
                mat[i][j] = float(t1)/float(t2)
        ax = sns.heatmap(mat,ax=axs[model_idx*3+idx], xticklabels=cpu_size ,yticklabels=gpu_size, cmap="coolwarm")
        # Rotate x-axis tick labels
        
        ax.set_xticklabels(cpu_size, rotation=0)

        # Rotate y-axis tick labels
        # ax.set_yticklabels(gpu_size, rotation=60)
        ax.invert_yaxis()
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f"OPT-{model}b/bsz:{bsz}")
fig.text(0.5, 0.04, 'CPU Cache Size', ha='center', va='center', fontsize=12)  # Common y label
fig.text(0.04, 0.5,  'GPU Cache Size', ha='center', va='center', rotation='vertical', fontsize=12)  # Common x label
 
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
gpu_name = "a6000"
plt.savefig(f'/home/wxd/code/Beyond/figures/hybrid-attn-speedup-{gpu_name}.pdf', bbox_inches='tight')
plt.show()