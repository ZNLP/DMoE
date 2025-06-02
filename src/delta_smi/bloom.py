from transformers import BloomForCausalLM
import torch
import torch.nn.functional as F
import sys

# obtain args
main_path = sys.argv[1]
model_name = sys.argv[2]

model_before_path = "bigscience/bloom-560m" if model_name == "bloom-560m" else "bigscience/bloom-1b7"
model_before = BloomForCausalLM.from_pretrained(model_before_path)

# obtain the vanilla parameter
param_before_ffns = []
for i in (21, 22, 23):
# for i in range(24):

    param_before_ffn_1 = model_before.transformer.h[i].mlp.dense_4h_to_h.weight.reshape(-1)
    param_before_ffn_2 = model_before.transformer.h[i].mlp.dense_h_to_4h.weight.reshape(-1)
    param_before_ffn = torch.hstack((param_before_ffn_1, param_before_ffn_2))
    param_before_ffns.append(param_before_ffn)

param_before = torch.hstack((param_before_ffns))
# print(param_before)
# print(param_before.size())

matrix = []
path = ["ar","bn","de","fr","hi","id","it","ja","ko","nl","ru","ta","te","th","uk","ur","vi","zh"]
for lang in path:
    print(lang)
    # extract the parameters after training
    if model_name == "bloom-560m":
        model_after_path = f"{main_path}/data/multi-18-10steps-bloom-560m/{lang}"
    elif model_name == "bloom-1.7b":
        model_after_path = f"{main_path}/data/multi-18-10steps-bloom-1.7b/{lang}"
    else:
        raise Exception(f"{model_name} must be bloom-560m or bloom-1.7b")

    model_after = BloomForCausalLM.from_pretrained(model_after_path)
    param_after_ffns = []
    for i in (21, 22, 23):
    # for i in range(24):

        param_after_ffn_1 = model_after.transformer.h[i].mlp.dense_4h_to_h.weight.reshape(-1)
        param_after_ffn_2 = model_after.transformer.h[i].mlp.dense_h_to_4h.weight.reshape(-1)
        param_after_ffn = torch.hstack((param_after_ffn_1, param_after_ffn_2))
        param_after_ffns.append(param_after_ffn)
    
    param_after = torch.hstack((param_after_ffns))

    # get the delta parameter during fine-tuning
    print(param_before==param_after)
    delta_param = param_before-param_after
    print(type(delta_param))
    print(delta_param)
    print(delta_param.size())
    matrix.append(delta_param)

matrix = torch.stack(matrix)
print(type(matrix))
print(matrix)
print(matrix.size())

cosine_sim = F.cosine_similarity(matrix.unsqueeze(1), matrix.unsqueeze(0), dim=-1)
cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
print(cosine_sim)

# convert the cosine similarity to the distance
cosine_dist = 1 - cosine_sim
print(cosine_dist)

if model_name == "bloom-560m":
    torch.save(cosine_sim, f'{main_path}/data/multi-18-10steps-bloom-560m-matrix/cosine_sim_10_last3.pt')
    torch.save(cosine_dist, f'{main_path}/data/multi-18-10steps-bloom-560m-matrix/cosine_dist_10_last3.pt')
elif model_name == "bloom-1.7b":
    torch.save(cosine_sim, f'{main_path}/data/multi-18-10steps-bloom-1.7b-matrix/cosine_sim_10_last3.pt')
    torch.save(cosine_dist, f'{main_path}/data/multi-18-10steps-bloom-1.7b-matrix/cosine_dist_10_last3.pt')
else:
    raise Exception(f"{model_name} must be bloom-560m or bloom-1.7b")