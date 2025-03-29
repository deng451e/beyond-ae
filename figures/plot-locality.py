import torch 
import numpy as np 
import matplotlib.pyplot as plt
cos_ = torch.nn.CosineSimilarity(dim=-1,eps = 1e-3)
fig, axs = plt.subplots(1, 2, figsize=(8, 3) )
axs = axs.flatten()
model="6.7"
path=f"/home/wxd/code/InfiniGen/wxd-test/wikitext-2-v1/opt-{model}b/512-1024-opt-{model}b.pth"
df = torch.load(path)
for layer_idx in [15]:
    
    layer_id = f"{layer_idx}-layer"
    for head in [0,4,25]:
        hold_temporal = []
        hold_spatial = []
        
        
        x = df[layer_id][256][head,0,:].float()

         
        for j in range(257,513):
            
            y = df[layer_id][j][head,0,:].float()
        
           
            ids = list(range(512))
            ids_rev = list(range(512))
           
            # x[x<torch.tensor(1e-3)] = torch.tensor(1e-3)
            # y[y<torch.tensor(1e-3)] = torch.tensor(1e-3)
            # _, x = torch.sort(x)  # Sorts in ascending order by default
            # _, y = torch.sort(y)  # Sorts in ascending order by default
            x = x.float()
            y = y.float()
            thre = 512
            
            hold_temporal.append(cos_(x[:thre],y[:thre]))
            hold_spatial.append(cos_(x[-257:],y[-257:]) )
          
       
        if head==15:    print(hold_spatial)
        axs[0].plot(list(range(257,513)),hold_temporal )
        
        axs[1].plot(list(range(257,513)),hold_spatial,label=f'head {head}')
        
        axs[0].set_xlabel("absolute kv position")
        axs[1].set_xlabel("relative kv position")
        # print(layer_idx,np.mean(hold_temporal),hold_temporal)
axs[0].spines[['right', 'top']].set_visible(False)  

axs[1].spines[['right', 'top']].set_visible(False)  

fig.legend(loc="upper right",ncol=3, bbox_to_anchor=(0.75, 1.05),fontsize=12)
fig.text(0.07, 0.5,  'attention similarity', ha='center', va='center', rotation='vertical', fontsize=12)  # Common x label
plt.savefig(f'/home/wxd/code/Beyond/figures/locality.pdf', bbox_inches='tight')
plt.show()