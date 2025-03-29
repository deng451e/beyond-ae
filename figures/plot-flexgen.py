import re
import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt

def read_log(f):
    with open(f, 'r') as file:
            log = file.read()
   
    # Define regex pattern to extract data
    pattern = re.compile(r'method:\s*(\w+).*?model:\s*([\w/.-]+).*?input:(\d+).*?output:(\d+).*?bzs:\s*(\d+).*?memory:([\d.e+-]+).*?Total:\s*([\d.]+).*?Prefill:\s*([\d.]+).*?Decode:\s*([\d.]+).*?Execution time:\s*(\d+)', re.S)

    # Parse the data
    results = []
    for match in pattern.finditer(log):
        
        results.append({
            "method": match.group(1),
            "model": match.group(2),
            "input": int(match.group(3)),
            "output": int(match.group(4)),
            "bzs": int(match.group(5)),
            "memory": float(match.group(6)),
            "total": float(match.group(7)),
            "prefill": float(match.group(8)),
            "decode": float(match.group(9)),
            "exec_time": int(match.group(10))
        })
    
    df = pd.DataFrame(results)
 
    # Convert numeric columns to appropriate types
    numeric_cols = ["input", "output", "bzs", "memory", "total", "prefill", "decode", "exec_time"]
    for col in numeric_cols:

        df[col] = pd.to_numeric(df[col])
    return df



def plot_save(df,name):
    labels = sorted(list(set(data["bzs"].values)))
    
    beyond_prefill = np.array(df[(df['method']=='beyond')]['prefill'] )
    beyond_decode =  np.array(df[(df['method']=='beyond')]['decode'] )
    beyond_mem =  np.array(df[(df['method']=='beyond')]['memory'] )
    
    infinigen_prefill = np.array(df[(df['method']=='infinigen')]['prefill'] )
    infinigen_decode = np.array(df[(df['method']=='infinigen')]['decode'] )
    infinigen_mem =  np.array(df[(df['method']=='infinigen')]['memory'] )
    
    flexgen_prefill = np.array(df[(df['method']=='flexgen')]['prefill'] )
    flexgen_decode = np.array(df[(df['method']=='flexgen')]['decode'] )
    flexgen_mem =np.array(df[(df['method']=='flexgen')]['memory'] )

    h2o_prefill = np.array(df[(df['method']=='h2o')]['prefill'] )
    h2o_decode =np.array(df[(df['method']=='h2o')]['decode'] )
    h2o_mem = np.array(df[(df['method']=='h2o')]['memory'] )
    
    cnt0,cnt1 =  len(infinigen_prefill),len(flexgen_prefill)
    labels = labels[:cnt1]
    if cnt0!=cnt1:
        add = [0 for _ in range(cnt1-cnt0)]
         
        infinigen_prefill = np.concatenate((infinigen_prefill,add))
        infinigen_decode = np.concatenate((infinigen_decode,add))
        infinigen_mem = np.concatenate((infinigen_mem,add))
   
    infinigen_mem  = infinigen_mem/flexgen_mem
    beyond_mem  = beyond_mem/flexgen_mem
    h2o_mem  = h2o_mem/flexgen_mem
    flexgen_mem  = flexgen_mem/flexgen_mem

    dims = [2**n for n in range(1,len(labels)+1)]
    bar_width = 0.2 # Width of the bars
    index = np.arange(len(dims))  # X locations for the bars
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs = axs.flatten()
    

    
    axs[0].bar(index  , beyond_prefill , bar_width , color='grey', edgecolor='black')
    axs[0].bar(index  , beyond_prefill + beyond_decode, bar_width, bottom=beyond_prefill , hatch='x',color='white', edgecolor='black', label='beyond')

    axs[0].bar(index + bar_width*1, infinigen_prefill , bar_width , color='grey', edgecolor='black')
    axs[0].bar(index + bar_width*1, infinigen_prefill + infinigen_decode, bar_width, bottom=infinigen_prefill   , hatch='\\'  , edgecolor='black' , color='black', label='infinigen')

    axs[0].bar(index + bar_width*2, flexgen_prefill , bar_width , color='grey', edgecolor='black' )
    axs[0].bar(index + bar_width*2, flexgen_prefill +flexgen_decode, bar_width, bottom=flexgen_prefill   , hatch='//',color='white', edgecolor='black', label='flexgen')


    axs[0].bar(index + bar_width*3, h2o_prefill , bar_width , color='grey' , edgecolor='black')
    axs[0].bar(index + bar_width*3, h2o_prefill + h2o_decode, bar_width, bottom=h2o_prefill ,color='white', edgecolor='black', label='h2o')

    
    axs[0].set_ylabel('Latency(sec)',fontsize=18)
    axs[0].set_xticks(index + bar_width / 0.65)
    axs[0].set_xticklabels(labels, ha='center',fontsize=18)
    # axs[0].set_title("vicuna-13b-v1.3 / mt_bench",pad=30,fontsize=18)
            
        

    # axs[0].set_yscale('log')
    
    axs[0].minorticks_off()
    axs[0].spines[['right', 'top']].set_visible(False)
            
    axs[1].bar(index, beyond_mem, bar_width ,  hatch='x',color='white', edgecolor='black')
    axs[1].bar(index+ bar_width*1,  infinigen_mem, bar_width , hatch='',color='black', edgecolor='black')
    axs[1].bar(index+ bar_width*2, flexgen_mem, bar_width   , hatch='//',color='white', edgecolor='black', ) 
    axs[1].bar(index+ bar_width*3, h2o_mem, bar_width, color='white', edgecolor='black')
    
    # axs[1].set_yscale('log')
    axs[1].set_ylabel('Normalized attn Memory',fontsize=18)
    axs[1].set_xticks(index + bar_width / 0.65)
    axs[1].set_xticklabels(labels, ha='center',fontsize=18)
    axs[1].set_ylim(0.9)  # Set y-axis limits for second plot

        
    axs[1].minorticks_off()
    axs[1].spines[['right', 'top']].set_visible(False)


    fig.legend(loc="upper right",ncol=4, bbox_to_anchor=(0.85, 1.15),fontsize=18)
    fig.text(0.53, 0.04, 'Batch Size' , ha='center', va='center' , fontsize=18) 
    plt.tight_layout()
    
    plt.savefig(f'/home/wxd/code/Beyond/figures/{name}.pdf', bbox_inches='tight')
    plt.show()


for model in ["opt-1.3b","opt-6.7b","opt-13b","opt-30b","opt-66b"]:
    f = f"/home/wxd/code/Beyond/evaluation/efficiency-test/flexgen-test/results/{model}-results.log"
    data = read_log(f)
    plot_save(data,model)