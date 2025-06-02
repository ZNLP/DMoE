from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import sys

# obtain args
main_path = sys.argv[1]

def extract_ffn_params(model, layer_idx):
    param_1 = model.base_model.layers[layer_idx].mlp.gate_proj.weight.reshape(-1)
    param_2 = model.base_model.layers[layer_idx].mlp.up_proj.weight.reshape(-1)
    param_3 = model.base_model.layers[layer_idx].mlp.down_proj.weight.reshape(-1)
    return torch.hstack((param_1, param_2, param_3))
    # return torch.hstack((param_2, param_3))

def extract_att_params(model, layer_idx):
    param_1 = model.base_model.layers[layer_idx].self_attn.q_proj.weight.reshape(-1)
    param_2 = model.base_model.layers[layer_idx].self_attn.k_proj.weight.reshape(-1)
    param_3 = model.base_model.layers[layer_idx].self_attn.v_proj.weight.reshape(-1)
    param_4 = model.base_model.layers[layer_idx].self_attn.o_proj.weight.reshape(-1)
    return torch.hstack((param_1, param_2, param_3, param_4))

delta_norm_results = {}

model_before_path = "Qwen/Qwen2.5-1.5B"
model_before = AutoModelForCausalLM.from_pretrained(model_before_path)
print(model_before.base_model)

plt.rcParams['font.size'] = 12

path = ["ar","bn","de","fr","hi","id","it","ja","ko","nl","ru","ta","te","th","uk","ur","vi","zh"]
for lang in path:
    model_after_path = f"{main_path}/data/multi-18-10steps-qwen2.5-1.5b/{lang}"
    model_after = AutoModelForCausalLM.from_pretrained(model_after_path)

    delta_norm_ffn = []
    delta_norm_att = []

    for i in tqdm(range(28), total=28, desc=f'lang:{lang}'):
        param_before_ffn = extract_ffn_params(model_before, i)
        param_before_att = extract_att_params(model_before, i)
        param_after_ffn = extract_ffn_params(model_after, i)
        param_after_att = extract_att_params(model_after, i)
        
        delta_ffn = param_after_ffn-param_before_ffn
        norm = torch.norm(delta_ffn).item()
        delta_norm_ffn.append(norm)
        delta_att = param_after_att-param_before_att
        norm = torch.norm(delta_att).item()
        delta_norm_att.append(norm)

    plt.figure()
    index = list(range(28))
    bar_width = 0.4
    plt.bar([x - bar_width / 2 for x in index], delta_norm_ffn, width=bar_width, color='deepskyblue', label='FFN')
    plt.bar([x + bar_width / 2 for x in index], delta_norm_att, width=bar_width, color='tomato', label='Attention')
    plt.xlabel('layer')
    t1 = "\parallel \Delta \\theta^{" + lang+ "} \parallel"
    plt.ylabel(r'${}$'.format(t1))
    plt.xticks(index)
    plt.legend(loc='upper center', ncols=2)

    plt.savefig(f'{main_path}/img/multi-18-10steps-qwen2.5-1.5b/delta-layer-data/{lang}.pdf') 
    plt.close()

    # save the results
    delta_norm_results[lang] = {
        "delta_norm_ffn": delta_norm_ffn,
        "delta_norm_att": delta_norm_att
    }

with open(f"{main_path}/img/multi-18-10steps-qwen2.5-1.5b/delta-layer-data/delta_norm_results.json", "w") as f:
    json.dump(delta_norm_results, f, indent=4)
