from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from constants import _HOME_DIR
import torch
import gemmoe
import bloommoe
import qwen2moe
import argparse

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

def copy_bloom_expert(
    src_path=f"{_HOME_DIR}/log/BloomMoEDyn/560M-EPS0.6-EXP6-S2_SD8/checkpoint-250",
    tgt_path=f"{_HOME_DIR}/data/BloomMoEDyn/560M-EPS0.6-EXP7",
    copy_expert_id=0,
):
    config = bloommoe.BloomMoEDynConfig.from_json_file(f"{src_path}/config.json")
    config.num_local_experts += 1

    me = bloommoe.BloomMoEDynForCausalLM(config)
    md = dict(bloommoe.BloomMoEDynForCausalLM.from_pretrained(src_path, torch_dtype=torch.bfloat16).named_parameters())
    tok = AutoTokenizer.from_pretrained(src_path, trust_remote_code=True)

    mes = dict(me.named_parameters())
    num_layer = me.config.num_hidden_layers

    res = {}

    # init the [transformer.word_embeddings.weight, transformer.word_embeddings_layernorm.weight, ..., transformer.h.x.mlp (dense layer), ...] of model
    for n in mes:
        if n in md:
            res[n] = md[n]

    for l in range(num_layer):
        is_moe = decode_booleans(config.moe_layer_index, num_layer)[l]
        if is_moe:
            # copy router weights
            gw = md[f"transformer.h.{l}.mlp.gate.weight"]
            res[f"transformer.h.{l}.mlp.gate.weight"] = torch.concatenate([gw, gw[copy_expert_id].reshape(1, -1)])

            # copy expert weights
            res[f"transformer.h.{l}.mlp.experts.{me.config.num_local_experts-1}.dense_h_to_4h.weight"] = md[f"transformer.h.{l}.mlp.experts.{copy_expert_id}.dense_h_to_4h.weight"]
            res[f"transformer.h.{l}.mlp.experts.{me.config.num_local_experts-1}.dense_h_to_4h.bias"] = md[f"transformer.h.{l}.mlp.experts.{copy_expert_id}.dense_h_to_4h.bias"]

            res[f"transformer.h.{l}.mlp.experts.{me.config.num_local_experts-1}.dense_4h_to_h.weight"] = md[f"transformer.h.{l}.mlp.experts.{copy_expert_id}.dense_4h_to_h.weight"]
            res[f"transformer.h.{l}.mlp.experts.{me.config.num_local_experts-1}.dense_4h_to_h.bias"] = md[f"transformer.h.{l}.mlp.experts.{copy_expert_id}.dense_4h_to_h.bias"]

    me.load_state_dict(res, strict=False)
    me.save_pretrained(tgt_path)
    tok.save_pretrained(tgt_path)
    return me, tok

def copy_gemma_expert(
    src_path=f"{_HOME_DIR}/log/GemMoEDyn/2B-EPS0.6-EXP6-S2_SD8/checkpoint-250",
    tgt_path=f"{_HOME_DIR}/data/GemMoEDyn/2B-EPS0.6-EXP6-ADD1",
    copy_expert_id=0,
):
    config = gemmoe.GemmoeDynConfig.from_json_file(f"{src_path}/config.json")
    config.num_local_experts += 1

    me = gemmoe.GemmoeDynForCausalLM(config)
    md = dict(gemmoe.GemmoeDynForCausalLM.from_pretrained(src_path, torch_dtype=torch.bfloat16).named_parameters())
    tok = AutoTokenizer.from_pretrained(src_path, trust_remote_code=True)

    mes = dict(me.named_parameters())
    num_layer = me.config.num_hidden_layers

    res = {}

    # init the [transformer.word_embeddings.weight, transformer.word_embeddings_layernorm.weight, ..., transformer.h.x.mlp (dense layer), ...] of model
    for n in mes:
        if n in md:
            res[n] = md[n]
        else:
            print(f"{n} is left.")

    for l in range(num_layer):
        is_moe = decode_booleans(config.moe_layer_index, num_layer)[l]
        if is_moe:
            # copy router weights
            gw = md[f"model.layers.{l}.layer.block_sparse_moe.gate.weight"]
            res[f"model.layers.{l}.layer.block_sparse_moe.gate.weight"] = torch.concatenate([gw, gw[copy_expert_id].reshape(1, -1)])

            # copy expert weights
            res[f"model.layers.{l}.layer.block_sparse_moe.experts.{me.config.num_local_experts-1}.w1.weight"] = md[f"model.layers.{l}.layer.block_sparse_moe.experts.{copy_expert_id}.w1.weight"]
            res[f"model.layers.{l}.layer.block_sparse_moe.experts.{me.config.num_local_experts-1}.w2.weight"] = md[f"model.layers.{l}.layer.block_sparse_moe.experts.{copy_expert_id}.w2.weight"]
            res[f"model.layers.{l}.layer.block_sparse_moe.experts.{me.config.num_local_experts-1}.w3.weight"] = md[f"model.layers.{l}.layer.block_sparse_moe.experts.{copy_expert_id}.w3.weight"]

    me.load_state_dict(res, strict=False)
    me.save_pretrained(tgt_path, safe_serialization=False)
    tok.save_pretrained(tgt_path)
    return me, tok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-type", type=str, default="gemma")
    parser.add_argument("-s", "--source-path", type=str, default=f"{_HOME_DIR}/log/GemMoEDyn/2B-EPS0.6-EXP6-S2_SD8/checkpoint-250")
    parser.add_argument("-i", "--copy-moe-index", type=int, default=0)
    parser.add_argument("-o", "--output-path", type=str, default=f"{_HOME_DIR}/data/GemMoEDyn/2B-EPS0.6-EXP6-ADD1")

    args = parser.parse_args()

    copy_expert_dict = {
        "bloom": copy_bloom_expert,
        "gemma": copy_gemma_expert,
    }

    copy_expert_dict[args.model_type](
        src_path=args.source_path,
        tgt_path=args.output_path,
        copy_expert_id=args.copy_moe_index,
    )
