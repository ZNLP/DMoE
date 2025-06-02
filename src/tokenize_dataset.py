import os
import transformers
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger

from datasets import load_dataset, load_metric, load_from_disk, DatasetDict, concatenate_datasets
from datasets import disable_caching
from datasets import Features, Sequence, Value

from torch.utils.data import Dataset, random_split
from itertools import chain
import argparse

def prepare_dataset(
        tokenizer,
        raw_datasets,
        tgt_path_in_disk,
        text_column_name:str="text",
        preprocessing_num_workers:int=32,
        block_size:int=4096,
        save=True,
    ):

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    column_names = raw_datasets[list(raw_datasets.keys())[0]].column_names
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    if block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        # without labels for saving space
        # result["labels"] = result["input_ids"].copy()
        return result
    
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("bool")),
    })

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        features=features,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    for key in lm_datasets.keys():
        lm_datasets[key] = lm_datasets[key].filter(lambda example: len(example['input_ids']) == block_size, num_proc=preprocessing_num_workers)

    if save:
        lm_datasets.save_to_disk(tgt_path_in_disk)
    
    return lm_datasets

if __name__ == "__main__":
    disable_caching()

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer", type=str, default="bigscience/bloom-560m", help="The path of tokenizer.",)
    parser.add_argument("-w", "--num-worker", type=int, default=16)
    parser.add_argument("-x", "--num-sharded", type=int, default=1)
    parser.add_argument("-b", "--block-size", type=int, default=4096)
    parser.add_argument("-r", "--random-shuffle-seed", type=int, default=-1)
    parser.add_argument("-c", "--cache-dir", type=str, default="~/.cache/huggingface/datasets")
    parser.add_argument("-n", "--text-column-name", type=str, default="text", help="The name of text column.",)
    parser.add_argument("-s", "--src-path", type=str, default="./data/source/3Groups-train", help="The path of source dataset.",)
    parser.add_argument("-j", "--is-jsonl-file", type=bool, default=False, help="Whether it is a jsonl file?",)
    parser.add_argument("-p", "--test-percentage", type=float, default=0.01, help="The percentage of test split file.",)
    parser.add_argument("-o", "--tgt-path", type=str, default="./data/datasets/3Groups-train", help="The path of target dataset.",)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, trust_remote_code=True)

    raw_datasets = load_from_disk(args.src_path) if not args.is_jsonl_file else load_dataset("json", data_files=args.src_path, cache_dir=args.cache_dir)["train"].train_test_split(args.test_percentage)
    
    if args.random_shuffle_seed > 0:
        raw_datasets = raw_datasets.shuffle(seed=args.random_shuffle_seed).flatten_indices()

    if args.num_sharded == 1:
        prepare_dataset(
            tokenizer=tokenizer,
            raw_datasets=raw_datasets,
            tgt_path_in_disk=args.tgt_path,
            text_column_name=args.text_column_name,
            preprocessing_num_workers=args.num_worker,
            block_size=args.block_size,
        )
    else:
        lens_dict = {k:int(len(raw_datasets[k])/args.num_sharded) for k in raw_datasets.keys()}

        for i in range(args.num_sharded):
            if i < 2:
                continue

            raw_dataset_i = DatasetDict()
            
            for k,v in raw_datasets.items():
                left_num = len(v) - lens_dict[k]*i
                raw_dataset_i[k] = v.select([(lens_dict[k]*i+j) for j in range(min(lens_dict[k], left_num))])
            
            processed_data = prepare_dataset(
                tokenizer=tokenizer,
                raw_datasets=raw_dataset_i,
                tgt_path_in_disk=args.tgt_path+f"_part{i}",
                text_column_name=args.text_column_name,
                preprocessing_num_workers=args.num_worker,
                block_size=args.block_size,
                save=True,
            )

            del processed_data
        
        processed_data_list = []
        for i in range(args.num_sharded):
            processed_data_list.append(load_from_disk(args.tgt_path+f"_part{i}"))
        
        processed_datasets = DatasetDict()

        for k in processed_data_list[0].keys():
            processed_datasets[k] = concatenate_datasets([d[k] for d in processed_data_list])
        
        processed_datasets.save_to_disk(args.tgt_path)
