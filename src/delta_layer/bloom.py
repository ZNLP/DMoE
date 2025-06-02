from transformers import BloomForCausalLM
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import sys

# obtain args
main_path = sys.argv[1]

delta_norm_results = {}

# model_before_path = "bigscience/bloom-560m"
model_before_path = "bigscience/bloom-1b7"
model_before = BloomForCausalLM.from_pretrained(model_before_path)
print(model_before.transformer)

plt.rcParams['font.size'] = 12

path = ["ar","bn","de","fr","hi","id","it","ja","ko","nl","ru","ta","te","th","uk","ur","vi","zh"]
for lang in path:
    model_after_path = f"{main_path}/data/multi-18-10steps-bloom-1.7b/{lang}"
    model_after = BloomForCausalLM.from_pretrained(model_after_path)

    delta_norm_ffn = []
    delta_norm_att = []

    for i in tqdm(range(24), total=24, desc=f'lang:{lang}'):
        param_before_ffn_1 = model_before.transformer.h[i].mlp.dense_4h_to_h.weight.reshape(-1)
        param_before_ffn_2 = model_before.transformer.h[i].mlp.dense_h_to_4h.weight.reshape(-1)
        param_before_ffn = torch.hstack((param_before_ffn_1, param_before_ffn_2))
        # print(param_before_ffn)
        # print(param_before_ffn.size())
        # norm = torch.norm(param_before_ffn).item()
        # print(norm)
        param_before_att_1 = model_before.transformer.h[i].self_attention.query_key_value.weight.reshape(-1)
        param_before_att_2 = model_before.transformer.h[i].self_attention.dense.weight.reshape(-1)
        param_before_att = torch.hstack((param_before_att_1, param_before_att_2))
        # print(param_before_att)
        # print(param_before_att.size())
        # norm = torch.norm(param_before_att).item()
        # print(norm)


        param_after_ffn_1 = model_after.transformer.h[i].mlp.dense_4h_to_h.weight.reshape(-1)
        param_after_ffn_2 = model_after.transformer.h[i].mlp.dense_h_to_4h.weight.reshape(-1)
        param_after_ffn = torch.hstack((param_after_ffn_1, param_after_ffn_2))
        # print(param_after_ffn)
        # print(param_after_ffn.size())
        # norm = torch.norm(param_after_ffn).item()
        # print(norm)
        param_after_att_1 = model_after.transformer.h[i].self_attention.query_key_value.weight.reshape(-1)
        param_after_att_2 = model_after.transformer.h[i].self_attention.dense.weight.reshape(-1)
        param_after_att = torch.hstack((param_after_att_1, param_after_att_2))
        # print(param_after_att)
        # print(param_after_att.size())
        # norm = torch.norm(param_after_att).item()
        # print(norm)


        delta_ffn = param_after_ffn-param_before_ffn
        # print(delta_ffn)
        # print(delta_ffn.size())
        norm = torch.norm(delta_ffn).item()
        # print(norm)
        delta_norm_ffn.append(norm)

        delta_att = param_after_att-param_before_att
        # print(delta_att)
        # print(delta_att.size())
        norm = torch.norm(delta_att).item()
        # print(norm)
        delta_norm_att.append(norm)

    plt.figure()
    index = list(range(24))
    bar_width = 0.4
    plt.bar([x - bar_width / 2 for x in index], delta_norm_ffn, width=bar_width, color='deepskyblue', label='FFN')
    plt.bar([x + bar_width / 2 for x in index], delta_norm_att, width=bar_width, color='tomato', label='Attention')
    plt.xlabel('layer')
    t1 = "\parallel \Delta \\theta^{" + lang+ "} \parallel"
    plt.ylabel(r'${}$'.format(t1))
    plt.xticks(index)
    plt.legend(loc='upper center', ncols=2)

    plt.savefig(f'{main_path}/img/multi-18-10steps-bloom-1.7b/delta-layer-data/{lang}.pdf')
    plt.close()

    # save the results
    delta_norm_results[lang] = {
        "delta_norm_ffn": delta_norm_ffn,
        "delta_norm_att": delta_norm_att
    }

with open(f"{main_path}/img/multi-18-10steps-bloom-1.7b/delta-layer-data/delta_norm_results.json", "w") as f:
    json.dump(delta_norm_results, f, indent=4)
