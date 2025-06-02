from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from constants import _HOME_DIR
import torch
import gemmoe
import bloommoe
import qwen2moe
import argparse

_3x2B_CONFIG = f"{_HOME_DIR}/data/GemMoE-Configs/gemmoe_3x2B.json"
_3x2B_DYN_CONFIG = f"{_HOME_DIR}/data/GemMoE-Configs/gemmoedyn_3x2B_IDX6.json"

_3x2B_BLOOMDYN_CONFIG = f"{_HOME_DIR}/data/BloomMoE-Configs/bloommoedyn_3x1.7B_IDX6.json"
_3x560M_BLOOMDYN_CONFIG = f"{_HOME_DIR}/data/BloomMoE-Configs/bloommoedyn_3x560M_IDX6.json"

_4xB5_QWEN2DYN_CONFIG = f"{_HOME_DIR}/data/Qwen2MoE-Configs/qwen2moedyn_4x0B5_IDX6.json"
_4x1B5_QWEN2DYN_CONFIG = f"{_HOME_DIR}/data/Qwen2MoE-Configs/qwen2moedyn_4x1B5_IDX6.json"

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

def single_prompt(model, tokenizer, prompt="Hello, I'm am conscious and", max_new_tokens:int=128, do_sample:bool=True, num_beams:int=1, top_k:int=50, top_p:float=0.95, temperature:float=0.7, cuda=True, verbose=False):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if cuda:
        model = model.cuda()
        input_ids = input_ids.cuda()
    with torch.inference_mode():
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams, top_k=top_k, top_p=top_p, temperature=temperature)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    if verbose:
        print(results)
    return results

def merge_bloom(
    moe_layer_index = 6,
    base_size = "560m",
    dense_paths = [
            "bigscience/bloom-560m",
            "bigscience/bloom-560m",
            "bigscience/bloom-560m",
        ],
    tgt_path = f"{_HOME_DIR}/data/BloomMoEDyn/3x1.7B-IDX6"
):
    # init from bloom-560m/1.7b and test the inference performance
    if base_size.lower() == "560m":
        config = bloommoe.BloomMoEDynConfig.from_json_file(_3x560M_BLOOMDYN_CONFIG)
    elif base_size.lower() == "1.7b":
        config = bloommoe.BloomMoEDynConfig.from_json_file(_3x2B_BLOOMDYN_CONFIG)
    else:
        raise Exception(f"Unseen model size {base_size.lower()}")

    # config.vocab_size = 250680
    config.num_local_experts = len(dense_paths)
    config.moe_layer_index = moe_layer_index
    config.output_router_logits = True

    me = bloommoe.BloomMoEDynForCausalLM(config)
    mds = [dict(AutoModelForCausalLM.from_pretrained(mp, trust_remote_code=True, torch_dtype=torch.bfloat16).named_parameters()) for mp in dense_paths]
    tok = AutoTokenizer.from_pretrained(dense_paths[0], trust_remote_code=True)

    mes = dict(me.named_parameters())
    num_layer = me.config.num_hidden_layers

    res = {}

    # init the [transformer.word_embeddings.weight, transformer.word_embeddings_layernorm.weight, ..., transformer.h.x.mlp (dense layer), ...] of model
    for n in mes:
        if n in mds[0]:
            res[n] = sum([md[n] for md in mds]) / len(mds)
    
    # construct the moe parameters
    for l in range(num_layer):
        is_moe = decode_booleans(config.moe_layer_index, num_layer)[l]
        if is_moe:
            for e in range(len(mds)):
                res[f"transformer.h.{l}.mlp.experts.{e}.dense_h_to_4h.weight"] = mds[e][f"transformer.h.{l}.mlp.dense_h_to_4h.weight"]
                res[f"transformer.h.{l}.mlp.experts.{e}.dense_h_to_4h.bias"] = mds[e][f"transformer.h.{l}.mlp.dense_h_to_4h.bias"]
                res[f"transformer.h.{l}.mlp.experts.{e}.dense_4h_to_h.weight"] = mds[e][f"transformer.h.{l}.mlp.dense_4h_to_h.weight"]
                res[f"transformer.h.{l}.mlp.experts.{e}.dense_4h_to_h.bias"] = mds[e][f"transformer.h.{l}.mlp.dense_4h_to_h.bias"]

    me.load_state_dict(res, strict=False)
    me.save_pretrained(tgt_path)
    tok.save_pretrained(tgt_path)
    return me, tok

def merge_gemma(
    moe_layer_index = 6,
    base_size = "2B",
    dense_paths = [
            "google/gemma-2b",
            "google/gemma-2b",
            "google/gemma-2b",
        ],
    tgt_path = f"{_HOME_DIR}/data/GemMoEDyn-3x2B-IDX6"
):
    if base_size.lower() == "2b":
        config = gemmoe.GemmoeDynConfig.from_json_file(_3x2B_DYN_CONFIG)
    else:
        raise Exception(f"Unseen model size {base_size.lower()}")

    config.moe_layer_index = moe_layer_index
    config.num_local_experts = len(dense_paths)
    config.output_router_logits = True

    me = gemmoe.GemmoeDynForCausalLM(config)
    mds = [dict(AutoModelForCausalLM.from_pretrained(mp, trust_remote_code=True, torch_dtype=torch.bfloat16).named_parameters()) for mp in dense_paths]
    tok = AutoTokenizer.from_pretrained(dense_paths[0], trust_remote_code=True)

    mes = dict(me.named_parameters())
    num_layer = me.config.num_hidden_layers

    res = {}
    # init the [model.embed_tokens.weight, model.norm.weight] of model
    for n in mes:
        if n in mds[0]:
            res[n] = sum([md[n] for md in mds]) / len(mds)
    
    # construct the moe parameters
    for l in range(num_layer):
        # Average the norm
        res[f"model.layers.{l}.layer.input_layernorm.weight"] = sum([md[f"model.layers.{l}.input_layernorm.weight"] for md in mds]) / len(mds)
        res[f"model.layers.{l}.layer.post_attention_layernorm.weight"] = sum([md[f"model.layers.{l}.post_attention_layernorm.weight"] for md in mds]) / len(mds)

        # Average the attention
        res[f"model.layers.{l}.layer.self_attn.q_proj.weight"] = sum([md[f"model.layers.{l}.self_attn.q_proj.weight"] for md in mds]) / len(mds)
        res[f"model.layers.{l}.layer.self_attn.k_proj.weight"] = sum([md[f"model.layers.{l}.self_attn.k_proj.weight"] for md in mds]) / len(mds)
        res[f"model.layers.{l}.layer.self_attn.v_proj.weight"] = sum([md[f"model.layers.{l}.self_attn.v_proj.weight"] for md in mds]) / len(mds)
        res[f"model.layers.{l}.layer.self_attn.o_proj.weight"] = sum([md[f"model.layers.{l}.self_attn.o_proj.weight"] for md in mds]) / len(mds)

        is_moe = decode_booleans(config.moe_layer_index, num_layer)[l]
        if is_moe:
            for e in range(len(mds)):
                res[f"model.layers.{l}.layer.block_sparse_moe.experts.{e}.w1.weight"] = mds[e][f"model.layers.{l}.mlp.gate_proj.weight"]
                res[f"model.layers.{l}.layer.block_sparse_moe.experts.{e}.w2.weight"] = mds[e][f"model.layers.{l}.mlp.down_proj.weight"]
                res[f"model.layers.{l}.layer.block_sparse_moe.experts.{e}.w3.weight"] = mds[e][f"model.layers.{l}.mlp.up_proj.weight"]
        else:
            # Adopt the first dense model
            res[f"model.layers.{l}.layer.mlp.w1.weight"] = mds[0][f"model.layers.{l}.mlp.gate_proj.weight"]
            res[f"model.layers.{l}.layer.mlp.w2.weight"] = mds[0][f"model.layers.{l}.mlp.down_proj.weight"]
            res[f"model.layers.{l}.layer.mlp.w3.weight"] = mds[0][f"model.layers.{l}.mlp.up_proj.weight"]

    me.load_state_dict(res, strict=False)
    # me.save_pretrained(tgt_path)
    me.save_pretrained(tgt_path, safe_serialization=False)
    tok.save_pretrained(tgt_path)
    return me, tok

def merge_qwen2(
    moe_layer_index = 6,
    base_size = "0.5B",
    dense_paths = [
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-0.5B",
        ],
    tgt_path = "./data/Qwen2MoEDyn/4x0.5B-IDX6"
):
    if base_size.lower() == "0.5b":
        config = qwen2moe.Qwen2moeDynConfig.from_json_file(_4xB5_QWEN2DYN_CONFIG)
    elif base_size.lower() == "1.5b":
        config = qwen2moe.Qwen2moeDynConfig.from_json_file(_4x1B5_QWEN2DYN_CONFIG)
    else:
        raise Exception(f"Unseen model size {base_size.lower()}")

    config.vocab_size = 151936
    config.num_experts = len(dense_paths)
    config.moe_layer_index = moe_layer_index
    config.output_router_logits = True

    me = qwen2moe.Qwen2moeDynForCausalLM(config)
    mds = [dict(AutoModelForCausalLM.from_pretrained(mp, trust_remote_code=True, torch_dtype=torch.bfloat16).named_parameters()) for mp in dense_paths]
    tok = AutoTokenizer.from_pretrained(dense_paths[0], trust_remote_code=True)

    mes = dict(me.named_parameters())
    num_layer = me.config.num_hidden_layers

    res = {}

    # init the [model.embed_tokens.weight, model.norm.weight, ..., model.layers.26.mlp (dense layer), ...] of model
    for n in mes:
        if n in mds[0]:
            res[n] = sum([md[n] for md in mds]) / len(mds)
    
    # construct the moe parameters
    for l in range(num_layer):
        is_moe = decode_booleans(config.moe_layer_index, num_layer)[l]
        if is_moe:
            for e in range(len(mds)):
                res[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = mds[e][f"model.layers.{l}.mlp.gate_proj.weight"]
                res[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = mds[e][f"model.layers.{l}.mlp.up_proj.weight"]
                res[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = mds[e][f"model.layers.{l}.mlp.down_proj.weight"]

    me.load_state_dict(res, strict=False)
    me.save_pretrained(tgt_path)
    tok.save_pretrained(tgt_path)
    return me, tok

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, default="bloom")
    parser.add_argument("-i", "--moe-layer-index", type=int, default=15728671)
    parser.add_argument("-s", "--model-size", type=str, default="560m")
    parser.add_argument("-p", "--model-paths", type=str, nargs='+', required=True)
    parser.add_argument("-o", "--output-path", type=str, default=f"{_HOME_DIR}/data/BloomMoEDyn/3x560M-IDX15728671")

    args = parser.parse_args()

    merge_dict = {
        "bloom": merge_bloom,
        "qwen2": merge_qwen2,
        "gemma": merge_gemma,
    }

    m0, tok = merge_dict[args.model_type](
        moe_layer_index = args.moe_layer_index,
        base_size = args.model_size,
        dense_paths = args.model_paths,
        tgt_path = args.output_path
    )

    # m0, tok = merge_bloom(
    #     moe_layer_index = 15728671,
    #     base_size = "560m",
    #     dense_paths=[
    #         f"bigscience/bloom-560m" for i in range(3)
    #     ],
    #     tgt_path=f"{_HOME_DIR}/data/BloomMoEDyn/3x560M-IDX15728671"
    # )

    # m0, tok = merge_qwen2(
    #     moe_layer_index = 16449567,
    #     base_size = "0.5B",
    #     dense_paths=[
    #         "Qwen/Qwen2.5-0.5B" for i in range(3)
    #     ],
    #     tgt_path=f"{_HOME_DIR}/data/Qwen2MoEDyn/3x0.5B-IDX16449567"
    # )

    # m0, tok = merge_gemma(
    #     moe_layer_index = 262143,
    #     base_size = "2B",
    #     dense_paths=[
    #         f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/ar-hi-ur/checkpoint-140",
    #         f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/bn-ta-te/checkpoint-140",
    #         f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/de-fr-nl/checkpoint-140",
    #         f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/id-it-ru/checkpoint-140",
    #         f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/ja-ko-zh/checkpoint-140",
    #         f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/th-uk-vi/checkpoint-140",
    #     ],
    #     tgt_path=f"{_HOME_DIR}/log/2b/2B-BTM-EXP6_SD1204096860/Merge-checkpoint-140"
    # )