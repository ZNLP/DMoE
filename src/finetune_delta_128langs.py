from transformers import BloomForCausalLM, BloomTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import sys
import os
from tqdm import tqdm

# ban wandb
os.environ["WANDB_DISABLED"] = "true"
device = torch.device('cuda')  # use default GPU
# obtain args
arg = sys.argv[1]  # language
local_model_path = sys.argv[2]  # model path
data_path = sys.argv[3]  # data path
output_dir = sys.argv[4]  # output path
layer_dir = sys.argv[5] # statistic path
# print input args 
print(f"Language: {arg}")
print(f"Model Path: {local_model_path}")
print(f"Data Path: {data_path}")
print(f"Output Directory: {output_dir}")
print(f"Layer Directory: {layer_dir}")

def extract_ffn_params(model, layer_idx):
    param_1 = model.transformer.h[layer_idx].mlp.dense_4h_to_h.weight.reshape(-1)
    param_2 = model.transformer.h[layer_idx].mlp.dense_h_to_4h.weight.reshape(-1)
    return torch.hstack((param_1, param_2))

def extract_att_params(model, layer_idx):
    param_1 = model.transformer.h[layer_idx].self_attention.query_key_value.weight.reshape(-1)
    param_2 = model.transformer.h[layer_idx].self_attention.dense.weight.reshape(-1)
    return torch.hstack((param_1, param_2))

# load bloom model and tokenizer
model = BloomForCausalLM.from_pretrained(local_model_path)
tokenizer = BloomTokenizerFast.from_pretrained(local_model_path)

param_before_ffn_list = []
param_before_att_list = []
for i in tqdm(range(24), total=24, desc=f'lang:{arg}'):
    # extract parameters
    param_before_ffn = extract_ffn_params(model, i)
    param_before_ffn_list.append(param_before_ffn)
    param_before_att = extract_att_params(model, i)
    param_before_att_list.append(param_before_att)

tokenized_train = Dataset.load_from_disk(data_path)
print(f"Loaded dataset with {len(tokenized_train)} blocks.")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # clm task
)


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=256,
    max_steps=10,
    learning_rate=8e-06,
    bf16=True,
    lr_scheduler_type="linear",
    save_strategy="no",
    eval_strategy="no",
    logging_steps=1,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train,
)

# start to train
trainer.train()

# task0: save the last three layer parameters for language clustering
layers_params = {}
for i in (21, 22, 23):
    param_after_ffn_1 = model.transformer.h[i].mlp.dense_4h_to_h.weight.reshape(-1)
    param_after_ffn_2 = model.transformer.h[i].mlp.dense_h_to_4h.weight.reshape(-1)
    layers_params[f"layer_{i}_4h_to_h"] = param_after_ffn_1
    layers_params[f"layer_{i}_h_to_4h"] = param_after_ffn_2
output_path = os.path.join(output_dir, f"{arg}.pt")
torch.save(layers_params, output_path)

# task1: plot delta
delta_norm_results = {}
delta_norm_ffn = []
delta_norm_att = []
for i in tqdm(range(24), total=24, desc=f'lang:{arg}'):
    param_after_ffn = extract_ffn_params(model, i)
    param_after_att = extract_att_params(model, i)

    param_after_ffn = param_after_ffn.to(device)
    param_after_att = param_after_att.to(device)

    param_before_ffn = param_before_ffn_list[i].to(device)  
    param_before_att = param_before_att_list[i].to(device)  

    delta_ffn = param_after_ffn - param_before_ffn
    norm_ffn = torch.norm(delta_ffn).item()
    delta_norm_ffn.append(norm_ffn)

    delta_att = param_after_att - param_before_att
    norm_att = torch.norm(delta_att).item()
    delta_norm_att.append(norm_att)

plt.figure()
index = list(range(24))
bar_width = 0.4
plt.bar([x - bar_width / 2 for x in index], delta_norm_ffn, width=bar_width, color='deepskyblue', label='delta_ffn')
plt.bar([x + bar_width / 2 for x in index], delta_norm_att, width=bar_width, color='tomato', label='delta_att')
plt.xlabel('layer')
plt.ylabel('delta')
plt.title('delta-layer')
plt.xticks(index)
plt.legend()

png_path = os.path.join(layer_dir, f"png/{arg}.png")
plt.savefig(png_path)
plt.close()

delta_norm_results[arg] = {
    "delta_norm_ffn": delta_norm_ffn,
    "delta_norm_att": delta_norm_att
}

layer_path = os.path.join(layer_dir, f"json/delta_norm_results_{arg}.json")
with open(layer_path, "w") as f:
    json.dump(delta_norm_results, f, indent=4)
