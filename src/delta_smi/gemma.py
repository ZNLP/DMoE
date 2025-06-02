from transformers import GemmaForCausalLM
import torch
import torch.nn.functional as F
import sys

# obtain args
main_path = sys.argv[1]

def extract_ffn_params(model, layer_idx):
    param_1 = model.base_model.layers[layer_idx].mlp.gate_proj.weight.reshape(-1)
    param_2 = model.base_model.layers[layer_idx].mlp.up_proj.weight.reshape(-1)
    param_3 = model.base_model.layers[layer_idx].mlp.down_proj.weight.reshape(-1)
    return torch.hstack((param_1, param_2, param_3))

model_before_path = "google/gemma-2b"
model_before = GemmaForCausalLM.from_pretrained(model_before_path)
print(model_before.base_model)

# obtain the vanilla parameter
param_before_ffns = []
for i in (15, 16, 17):
    param_before_ffn = extract_ffn_params(model_before, i)
    param_before_ffns.append(param_before_ffn)
param_before = torch.hstack((param_before_ffns))

matrix = []
path = ["ar","bn","de","fr","hi","id","it","ja","ko","nl","ru","ta","te","th","uk","ur","vi","zh"]
for lang in path:
    print(lang)
    # extract the parameters after training
    model_after_path = f"{main_path}/data/multi-18-10steps-gemma-2b/{lang}"
    model_after = GemmaForCausalLM.from_pretrained(model_after_path)
    param_after_ffns = []
    for i in (15, 16, 17):
        param_after_ffn = extract_ffn_params(model_after, i)
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
cosine_sim[0][0] = 1.0000
print(cosine_sim)
torch.save(cosine_sim, f'{main_path}/data/multi-18-10steps-gemma-2b-matrix/cosine_sim_10_last3.pt')

# convert the cosine similarity to the distance
cosine_dist = 1 - cosine_sim
print(cosine_dist)
torch.save(cosine_dist, f'{main_path}/data/multi-18-10steps-gemma-2b-matrix/cosine_dist_10_last3.pt')
