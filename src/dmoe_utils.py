import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    BitsAndBytesConfig,
    TrainingArguments,
)

import itertools

import json
from bloommoe import BloomMoEDynConfig, BloomMoEDynForCausalLM
from gemmoe import GemmoeDynConfig, GemmoeDynForCausalLM
from qwen2moe import Qwen2moeDynConfig, Qwen2moeDynForCausalLM

_CLM_CLASS_DICT = {
    "bloommoedyn": BloomMoEDynForCausalLM,
    "gemmoedyn": GemmoeDynForCausalLM,
    "qwen2moedyn": Qwen2moeDynForCausalLM,
}

_CONFIG_CLASS_DICT = {
    "bloommoedyn": BloomMoEDynConfig,
    "gemmoedyn": GemmoeDynConfig,
    "qwen2moedyn": Qwen2moeDynConfig,
}

class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=4096,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
        start_idx=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        if start_idx != 0:
            print(f"Reset the start index of dataset to {start_idx}.")
            self.dataset = concatenate_datasets([dataset.select(range(start_idx, len(dataset))), dataset.select(range(0, start_idx))])
        self.need_tokenize = False if 'input_ids' in dataset.features else True
        self.content_field = 'input_ids' if 'input_ids' in dataset.features else content_field
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token
    
    def direct_iter(self):
        iterator = iter(self.dataset)
        while True:
            buffer, buffer_len = [], 0

            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    sample = next(iterator)[self.content_field]
                    assert len(sample) == self.seq_length
                    buffer.append(sample)
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break

            examples = buffer

            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

    def token_iter(self):
        iterator = iter(self.dataset)
        while True:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
    
    def multiprocessing_iter(self):
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id
        iterator = self.token_iter() if self.need_tokenize else self.direct_iter()

        # return itertools.islice(map(self.sample_mapper, iterator), worker_id, None, worker_total_num)
        return itertools.islice(iterator, worker_id, None, worker_total_num)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is None:
            return self.token_iter() if self.need_tokenize else self.direct_iter()
        else:
            return self.multiprocessing_iter()


class MultilingualConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        datasets,
        infinite=False,
        seq_length=1024,
        num_of_sequences=4096,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
        start_idx=0,
        random_seed=0,
        consecutive_num=128, # define the num of consecutive samples
        name2id=None,
    ):
        random.seed(random_seed)

        self.id2len = {}
        self.id2name = {}
        self.id2dataset = {}
        self.name2id = {}
        self.ids = []
        self.consecutive_num = consecutive_num
        
        if name2id is None:
            for i, (k, v) in enumerate(datasets.items()):
                self.id2name[i] = k
                self.name2id[k] = i
                self.id2len[i] = len(v)
                self.ids.extend([i for _ in range((len(v) // consecutive_num)+1)])
        else:
            for k, i in name2id.items():
                v = datasets[k]
                i = int(i)
                self.id2name[i] = k
                self.name2id[k] = i
                self.id2len[i] = len(v)
                self.ids.extend([i for _ in range((len(v) // consecutive_num)+1)])

        print(f"Current expert id to dataset mapping is:\n{self.id2name}")

        random.shuffle(self.ids)

        self.start_idx_dict = {k:0 for k in self.id2name.keys()}
        self.curr_id = start_idx // consecutive_num

        for p in range(self.curr_id):
            self.start_idx_dict[self.ids[p]] += consecutive_num        
        self.start_idx_dict[self.ids[self.curr_id]] += (start_idx % consecutive_num)

        self.curr_num = start_idx % consecutive_num
        
        for i, (k, v) in enumerate(datasets.items()):
            self.id2dataset[self.name2id[k]] = ConstantLengthDataset(
                tokenizer=tokenizer,
                dataset=v,
                infinite=infinite,
                seq_length=seq_length,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                content_field=content_field,
                shuffle=shuffle,
                add_eos_token=add_eos_token,
                start_idx=self.start_idx_dict[self.name2id[k]],
            )
    
    def single_thread_iter(self):
        self.id2iter = {}
        for i, v in self.id2dataset.items():
            self.id2iter[i] = iter(v)

        while True:            
            if self.curr_num >= self.consecutive_num:
                self.curr_num -= self.consecutive_num
                self.curr_id += 1
            if self.curr_id >= len(self.ids):
                self.curr_id -= len(self.ids)                
            self.curr_iter = self.id2iter[self.ids[self.curr_id]]
            self.curr_num += 1
            example = next(self.curr_iter)
            example["dataset_id"] = torch.LongTensor([self.ids[self.curr_id]])

            yield example

    def multi_thread_iter(self):
        worker_total_num = torch.utils.data.get_worker_info().num_workers
        worker_id = torch.utils.data.get_worker_info().id
        iterator = iter(self.single_thread_iter())

        return itertools.islice(iterator, worker_id, None, worker_total_num)
    
    def __iter__(self):
        if torch.utils.data.get_worker_info() is None:
            return self.single_thread_iter()
        else:
            return self.multi_thread_iter()

def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column])['input_ids'])
        # total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def load_processed_dataset(tokenizer, args):
    dataset = load_from_disk(args.dataset_name)
    train_data = dataset["train"]
    valid_data = dataset["test"] if "test" in dataset else dataset["validation"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field) if 'text' in train_data.features else 1
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=True,
        add_eos_token=False,
        start_idx=args.train_start_idx,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=False,
        add_eos_token=False,
    )

    return train_dataset, valid_dataset

def load_expert_mapping_datasets(tokenizer, args):
    dataset = load_from_disk(args.dataset_name)
    
    print(f"Split the dataset into train({1 - args.eval_data_ratio:.2f}), valid({args.eval_data_ratio:.2f}) ...")
    train_data = DatasetDict()
    valid_data = DatasetDict()
    train_size_dict = {}
    eval_size_dict = {}
    features = None
    for k, v in dataset.items():
        ds = v.train_test_split(test_size=args.eval_data_ratio)
        train_data[k] = ds["train"]
        train_size_dict[k] = len(ds["train"])
        valid_data[k] = ds["test"]
        eval_size_dict[k] = len(ds["test"])
        features = v.features
    
    print(f"Size of the train set: {train_size_dict}. Size of the validation set: {eval_size_dict}")
    chars_per_token = chars_token_ratio(train_data, tokenizer, args.dataset_text_field) if 'text' in features else 1
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    dataset_key2id = None
    if len(args.dataset_key2id) > 0:
        with open(args.dataset_key2id, "r") as f:        
            dataset_key2id = json.load(f)

    train_dataset = MultilingualConstantLengthDataset(
        tokenizer,
        datasets=train_data,
        infinite=True,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=True,
        add_eos_token=False,
        start_idx=args.train_start_idx,
        random_seed=args.seed,
        consecutive_num=args.consecutive_num,
        name2id=dataset_key2id
    )

    valid_dataset = MultilingualConstantLengthDataset(
        tokenizer,
        datasets=valid_data,
        infinite=False,
        seq_length=args.max_seq_length,
        chars_per_token=chars_per_token,
        content_field=args.dataset_text_field,
        shuffle=False,
        add_eos_token=False,
        random_seed=args.seed,
        consecutive_num=args.consecutive_num,
        name2id=dataset_key2id
    )

    return train_dataset, valid_dataset

def create_datasets(tokenizer, args):
    if args.map_to_expert:
        return load_expert_mapping_datasets(tokenizer, args)
    else:
        return load_processed_dataset(tokenizer, args)

def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}
    
    model_config, _ = PretrainedConfig.get_config_dict(args.model_name)

    clm_class = _CLM_CLASS_DICT[model_config['model_type']] if model_config['model_type'] in _CLM_CLASS_DICT else AutoModelForCausalLM

    model = clm_class.from_pretrained(
        args.model_name,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if args.router_aux_loss_coef >= 0:
        model.router_aux_loss_coef = args.router_aux_loss_coef
        model.config.router_aux_loss_coef = args.router_aux_loss_coef

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path is not None else args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model.resize_token_embeddings(len(tokenizer))

    return model, peft_config, tokenizer


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

def encode_booleans(bool_lst):
    """
    Convert the index(bool) of whether layer is dense or not into ``moe_layer_index`` in binary.
    For example: [False, False, True](left to right) -> 100 (binary, right to left) -> 4
    """
    res = 0
    for i, bval in enumerate(bool_lst):
        res += int(bval) << i
    return res

def decode_booleans(intval, bits):
    """
    Convert the binary ``moe_layer_index`` into the bool array.
    """
    res = []
    for bit in range(bits):
        mask = 1 << bit
        res.append((intval & mask) == mask)
    return res
