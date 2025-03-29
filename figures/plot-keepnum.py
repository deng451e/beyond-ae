import torch 
import numpy as np 
import matplotlib.pyplot as plt
model="6.7"
path=f"/home/wxd/code/InfiniGen/wxd-test/wikitext-2-v1/opt-{model}b/512-1024-opt-{model}b.pth"
df = torch.load(path)
for layer_idx in [15]:
    alpha = 0.1
    layer_id = f"{layer_idx}-layer"
    A = df[layer_id][1].squeeze()
    keep_nums = torch.sum(A >= (1/A.size(-1)*alpha)  ,dim=-1) /A.size(-1)
    keep_nums = keep_nums.numpy()
    indices = range(len(keep_nums))
    plt.figure(figsize=(8, 3))
    plt.bar(indices, keep_nums,color='white', edgecolor='black')
    plt.xlim(-0.5,31.5)
    s = torch.sum(A[A >= (1/A.size(-1)*alpha)])/A.size(0)
    plt.xlabel('Heads',fontsize=15)
    plt.ylabel('Percent of Selected KV',fontsize=15)
    plt.title(f'Avg attn weight sum across heads: {s:.2} ')

    y_value = np.mean(keep_nums)
    # Add label for the line
    plt.axhline(y=y_value, color='red', linestyle='--', label=f'y = {y_value:.2}')
    plt.text(x=0.5, y=y_value + 0.1, s=f'y = {y_value:.2}', color='red',fontsize=15)


    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/home/wxd/code/Beyond/figures/keepnum-opt6.7b.pdf', bbox_inches='tight')
    plt.show()