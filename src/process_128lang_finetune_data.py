from transformers import BloomForCausalLM, BloomTokenizerFast
from datasets import Dataset
import sys
from constants import _HOME_DIR

lang = sys.argv[1]
# output language
print(f"Language: {lang}")

local_model_path = f"bigscience/bloom-1b7"
model = BloomForCausalLM.from_pretrained(local_model_path)
tokenizer = BloomTokenizerFast.from_pretrained(local_model_path)

train_file_path = f"{_HOME_DIR}/data/MADLAD-400/{lang}"
train_dataset = Dataset.load_from_disk(train_file_path)
print(train_dataset)
print(len(train_dataset))
if len(train_dataset) > 102400:
    train_dataset = train_dataset.select(range(102400))
print(len(train_dataset))

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False, padding=False)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
print(len(tokenized_train['input_ids']))

all_input_ids = []
for index, item in enumerate(tokenized_train):
    all_input_ids.extend(item['input_ids'])
    if len(all_input_ids) >= 10485760:
        print(index)
        break
print(len(all_input_ids))

if len(all_input_ids) > 10485760:
    print(len(all_input_ids))
    all_input_ids = all_input_ids[0:10485760]
    
    block_size = 512
    blocks = [all_input_ids[i:i + block_size] for i in range(0, len(all_input_ids), block_size)]

    attention_masks = []
    for block in blocks:
        attention_masks.append([1] * len(block))

    final_dataset = Dataset.from_dict({
        "input_ids": blocks,
        "attention_mask": attention_masks
    })

    print(f"Processed {lang} dataset with {len(final_dataset)} blocks, each containing up to {block_size} tokens.")

    save_path = f"{_HOME_DIR}/data/MADLAD-400-for-Cluster/{lang}"
    final_dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")
else:
    print(f"The data of *{lang}* is not enough.")
