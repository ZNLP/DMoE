import json
import numpy as np
import argparse

def normalize(arr:list):
    d_min = np.min(arr)
    d_max = np.max(arr)
    return [(x-d_min)/(d_max-d_min) for x in arr]

def encode_booleans(bool_lst):
    """
    Convert the index(bool) of whether layer is dense or not into ``moe_layer_index`` in binary.
    For example: [False, False, True](left to right) -> 100 (binary, right to left) -> 4
    """
    res = 0
    for i, bval in enumerate(bool_lst):
        res += int(bval) << i
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--statistic-file", type=str, default="./bloom560m.json")
    parser.add_argument("-e", "--epsilon", type=float, default=0.6)

    args = parser.parse_args()

    stat_file = args.statistic_file

    epsilon = args.epsilon

    with open(stat_file, "r") as f:
        data = json.load(f)

    langs = list(data.keys())

    num_layer = len(data[langs[0]]["delta_norm_ffn"])
    # num_layer = len(data[langs[0]]["delta_norm_att"])

    layer_ffn_avg_stat = []

    for l in range(num_layer):
        delta_ffns = [data[lang]["delta_norm_ffn"][l] for lang in langs]
        delta_atts = [data[lang]["delta_norm_att"][l] for lang in langs]

        delta_ffns_avg = np.mean(delta_ffns)
        delta_ffns_std = np.std(delta_ffns)

        delta_atts_avg = np.mean(delta_atts)
        delta_atts_std = np.std(delta_atts)

        layer_ffn_avg_stat.append(delta_ffns_avg)

        # print(f"{l},{delta_ffns_avg:.8f},{delta_ffns_std:.8f},{delta_atts_avg:.8f},{delta_atts_std:.8f}")
    
    layer_ffn_avg_stat = normalize(layer_ffn_avg_stat)

    is_moe = [l>= epsilon for l in layer_ffn_avg_stat]

    print(f"Final index for epsilon={epsilon} is {encode_booleans(is_moe)}")
    print("Details:")
    for l in range(num_layer):
        print(f"Layer-{l}\t({layer_ffn_avg_stat[l]:.4f}): {'MoE' if is_moe[l] else 'Dense'}")