from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from constants import _HOME_DIR
import torch
import torch.nn.functional as F
import sys
import os

# ban wandb
os.environ["WANDB_DISABLED"] = "true"
# obtain args
arg = sys.argv[1]
local_model_path = sys.argv[2]
model_name = sys.argv[3].lower()

# print input args 
print(f"Language: {arg}")
print(f"local_model_path: {local_model_path}")

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# load local json data
train_file_path = f"{_HOME_DIR}/data/source/CulturaX/{arg}.json"
# train_file_path = f"{_HOME_DIR}/data/source/MADLAD-400/{arg}.json"

train_dataset = load_dataset('json', data_files=train_file_path, split='train')
train_dataset = train_dataset.select(range(30000))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
tokenized_train = train_dataset.map(tokenize_function, batched=True)

columns_to_remove = ['timestamp', 'source', 'text', 'url']
tokenized_train = tokenized_train.remove_columns(columns_to_remove)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # clm task
)

training_args = TrainingArguments(
    output_dir=f"{_HOME_DIR}/data/multi-18-10steps-{model_name}/{arg}",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=256,
    max_steps=10,
    learning_rate=8e-06,
    bf16=True,
    lr_scheduler_type="linear",
    save_strategy="epoch",
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

# save models and tokenizer
output_dir=f"{_HOME_DIR}/data/multi-18-10steps-{model_name}/{arg}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
