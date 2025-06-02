from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from datasets import load_dataset, load_from_disk
from bloommoe import BloomMoEDynConfig, BloomMoEDynForCausalLM
from gemmoe import GemmoeDynConfig, GemmoeDynForCausalLM
from qwen2moe import Qwen2moeDynConfig, Qwen2moeDynForCausalLM
from tqdm import tqdm
import random
import torch
import argparse
import json

_CLM_CLASS_DICT = {
    "bloommoedyn": BloomMoEDynForCausalLM,
    "gemmoedyn": GemmoeDynForCausalLM,
    "qwen2moedyn": Qwen2moeDynForCausalLM,
}

def eval_ppl(
    model,
    encodings,
    dataset_id = None,
    max_length = 2048,
    stride = 1024,
):
    # max_length = model.config.n_positions
    # stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    # print(seq_len)
    # print(stride)
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids) if dataset_id is None else model(input_ids, labels=target_ids, dataset_id=torch.tensor([dataset_id], device=device, dtype=torch.int64))

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss.to(torch.float32)

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll).item()
    return ppl, n_tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="./data/bloom-560m")
    parser.add_argument("-d", "--json-dataset", type=str, default="./data/source/9Groups-test")
    parser.add_argument("-l", "--language-name", type=str, default="en")
    parser.add_argument("-n", "--num-block", type=int, default=1000)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--output-file", type=str)

    args = parser.parse_args()

    device="cuda:0"

    model_config, _ = PretrainedConfig.get_config_dict(args.model_path)

    clm_class = _CLM_CLASS_DICT[model_config['model_type']] if model_config['model_type'] in _CLM_CLASS_DICT else AutoModelForCausalLM

    model = clm_class.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2" if "bloom" not in model_config['model_type'] else "eager", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if hasattr(model.config, "output_router_logits"):
        # reduce the aux_loss for moe
        model.config.output_router_logits = False

    test = load_from_disk(args.json_dataset)[args.language_name.lower()]
    min_len = len(test)
    random.seed(args.seed)
    ids = [i for i in range(min_len)]
    random.shuffle(ids)
    rand_ids = ids[:args.num_block]
    test_k = test.select(rand_ids)
    encodings = tokenizer("\n\n".join(test_k["text"]), return_tensors="pt")

    res = {}
    is_dmoe = hasattr(model.config, "num_local_experts")
    min_expert = -1
    min_ppl = -1
    for ei in tqdm(range(getattr(model.config, "num_local_experts", 1))):
        ppl, n_tokens = eval_ppl(
            model=model,
            encodings=encodings,
            max_length=2048,
            stride=1024,
            dataset_id = ei if is_dmoe else None
        )

        if min_expert < 0 or ppl < min_ppl:
            min_expert = ei
            min_ppl = ppl

        res[ei] = {"perplexity": ppl, "num_tokens": n_tokens}

    print(f"The best expert is expert-{min_expert}(ppl: {min_ppl})")
    
    with open(args.output_file, "w") as f:
        json.dump(res, f, indent="\t")
