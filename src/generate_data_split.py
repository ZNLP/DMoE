import os
from transformers import AutoTokenizer
from constants import _HOME_DIR
from datasets import load_from_disk, load_dataset, concatenate_datasets, DatasetDict, disable_caching
import json
from tqdm import tqdm

_LANGS18=["ru", "uk", "de", "nl", "fr", "it", "ar", "ta", "te", "hi", "bn", "ur", "vi", "id", "ja", "ko", "th", "zh"]

_NEWLANGS4=["be", "sr", "ml", "mr"]

_MADLAD400_STAT_PATH=f"{_HOME_DIR}/data/MADLAD-400/madlad400_metadata.csv"
_MADLAD400_DATA_PATH="{dir}/data/MADLAD-400/{lang}"

# all data path
# _CULTURAX_DOC_STAT=f"{_HOME_DIR}/data/source/CulturaX/all_langs_metadata.csv"
# _CULTURAX_FILE_PATH="{_HOME_DIR}/data/source/CulturaX/{lang}.json"

# 1k mini test samples for demo
_CULTURAX_DOC_STAT=f"{_HOME_DIR}/data/source/CulturaX-1k-samples/metadata.csv"
_CULTURAX_FILE_PATH="{dir}/data/source/CulturaX-1k-samples/{lang}.json"

def utf8len(s):
    return len(s.encode('utf-8'))

def read_csv(path):
    res = {}
    with open(path, "r") as f:
        for i, l in enumerate(f.readlines()):
            if i == 0:
                continue
            lang, num_line = l.strip().split(",")
            res[lang] = int(num_line)
    return res
            
def char_line_tok_rate(data, tok, num:int=1000):
    num_char = 0
    num_tok = 0
    diter = iter(data)
    num_line = min(len(data), num)
    for i in range(num_line):
        s = next(diter)['text']
        num_char += utf8len(s)
        toks = tok.tokenize(s)
        num_tok += len(toks)
    # tok/char, tok/line
    return num_tok/num_char, num_tok/num_line

def num_lines_dist(
    langs: list,
    total_token_amount: int,
    size_dict: dict,
    tok_line_rate_dict: dict,
    sample_alpha:float=0.3,
) -> dict:
    size_sum = sum([size_dict[l]**sample_alpha for l in langs])
    lang_line_dict = {}
    for lang in langs:
        token_size = ((size_dict[lang]**sample_alpha) / size_sum) * total_token_amount
        tok_line_rate = tok_line_rate_dict[lang]
        lang_line = int(token_size / tok_line_rate) + 1
        lang_line_dict[lang] = lang_line
    return lang_line_dict
        
def extract_data(
    lang_line_dict: dict,
    json_path_format: str=None,
    dataset_path_format: str=None,
    test_portion: float = -1,
    seed=0
):
    langs = list(lang_line_dict.keys())
    all_data = []
    all_test_datasets = []
    for lang in tqdm(langs, desc="Loading data of some languages ..."):
        lang_dataset = load_dataset("json", data_files={"train":json_path_format.format(dir=_HOME_DIR,lang=lang)})["train"] if json_path_format is not None else load_from_disk(dataset_path_format.format(dir=_HOME_DIR, lang=lang))
        lang_line = lang_line_dict[lang]

        # check madlad-400 corpus
        if dataset_path_format is not None:
            lang_total_line_dict = read_csv(_MADLAD400_STAT_PATH)
            if lang_line > len(lang_dataset) and len(lang_dataset) < lang_total_line_dict[lang]:
                print(f"!!! Please download more data for {lang}(needs:{lang_line}, current: {len(lang_dataset)}/{lang_total_line_dict[lang]})")
                raise Exception(f"!!! Please download more data for {lang}(needs:{lang_line}, current: {len(lang_dataset)}/{lang_total_line_dict[lang]})")

        if test_portion > 0:
            num_test = min(lang_line, len(lang_dataset)) * test_portion
            split_dataset = lang_dataset.train_test_split(num_test/len(lang_dataset))
            lang_dataset = split_dataset['train']
            test_dataset = split_dataset['test']
            all_test_datasets.append((lang, test_dataset))
        repeat = int(lang_line/ len(lang_dataset)) + int(lang_line % len(lang_dataset) > 0)
        if repeat > 1:
            print(f"Data of {lang} is repeat for {repeat} times ({lang_line} from {len(lang_dataset)})")
            lang_dataset = concatenate_datasets([lang_dataset for i in range(repeat)])
        lang_dataset = lang_dataset.shuffle(seed=seed)
        all_data.append(lang_dataset.select([i for i in range(lang_line)]))
    return concatenate_datasets(all_data).shuffle(seed=seed).flatten_indices(), all_test_datasets

def merge_raw_data(
    lang_groups: dict,
    lang_size_dict: dict,
    tok_line_rate_dict: dict,
    save_path,
    test_path,
    total_tokens = 1500000000,
    sample_alpha = 0.3,
    test_portion = 0.01,
    is_culturax = True,
):
    res_dict = DatasetDict()

    all_test_data = []
    for lang_group in lang_groups.values():
        line_dict = num_lines_dist(
            langs=list(lang_group),
            total_token_amount=total_tokens,
            size_dict=lang_size_dict,
            tok_line_rate_dict=tok_line_rate_dict,
            sample_alpha=sample_alpha,
        )

        lang_dataset, test_datasets = extract_data(
                lang_line_dict=line_dict,
                json_path_format=_CULTURAX_FILE_PATH if is_culturax else None,
                dataset_path_format=_MADLAD400_DATA_PATH if not is_culturax else None,
                test_portion=test_portion
            )
        
        res_dict["-".join(list(lang_group))] = lang_dataset
        # res_dict["128langs"] = lang_dataset
        all_test_data.extend(test_datasets)
    
    res_dict.save_to_disk(save_path)

    test_dataset = DatasetDict()
    for (lang, dl) in all_test_data:
        test_dataset[lang] = dl
    test_dataset.save_to_disk(test_path)

def generate_culturax(
    tok,
    lang_groups = {
        0: ('ar', 'bn', 'hi', 'id', 'ta', 'ur'),
        1: ('de', 'fr', 'it', 'nl', 'vi', 'zh'),
        2: ('ja', 'ko', 'ru', 'te', 'th', 'uk'),
    },
    save_path = f"{_HOME_DIR}/data/source/3Groups-train",
    test_path = f"{_HOME_DIR}/data/source/3Groups-test",
    total_tokens=1500000000,
    sample_alpha=0.3,
    test_portion=0.01
):

    tok_char_rate_dict = {}
    tok_line_rate_dict = {}
    for lang in tqdm(_LANGS18, desc="Count tokens ..."):
    # for lang in tqdm(_NEWLANGS4, desc="Count tokens ..."):
        data = load_dataset("json", data_files={"train":_CULTURAX_FILE_PATH.format(dir=_HOME_DIR, lang=lang)})["train"]
        tok_char_rate, tok_line_rate = char_line_tok_rate(data, tok, num=500)
        tok_char_rate_dict[lang] = tok_char_rate
        tok_line_rate_dict[lang] = tok_line_rate
        print(f"{lang}\t{tok_char_rate:.4f}(tok/char)\t{tok_line_rate:.4f}(tok/line)")
    
    lang_size_dict = {}
    lang_all_line_dict = read_csv(_CULTURAX_DOC_STAT)
    for lang in _LANGS18:
    # for lang in _NEWLANGS4:
        lang_size_dict[lang] = int(lang_all_line_dict[lang] * (tok_line_rate_dict[lang] / tok_char_rate_dict[lang]))
    
    
    merge_raw_data(
        lang_groups=lang_groups,
        lang_size_dict=lang_size_dict,
        tok_line_rate_dict=tok_line_rate_dict,
        save_path=save_path,
        test_path=test_path,
        total_tokens=total_tokens,
        sample_alpha=sample_alpha,
        test_portion=test_portion
    )


def generate_madlad(
    tok,
    lang_groups = {
        0: ('ceb', 'hil', 'fil', 'ilo', 'en', 'so', 'la', 'lg'), 
        1: ('xh', 'zu', 'sw', 'sn', 'ny', 'fr', 'it', 'yo'), 
        2: ('am', 'ti', 'my', 'ka', 'dv', 'lo', 'he', 'ko'), 
        3: ('es', 'pt', 'ca', 'vi', 'ar', 'id', 'eu', 'hi'), 
        4: ('cnh', 'lus', 'kha', 'ha', 'uz', 'de', 'nl', 'cs'), 
        5: ('ky', 'tyv', 'sah', 'tt', 'mn', 'kk', 'ru', 'kaa'), 
        6: ('da', 'no', 'fo', 'is', 'et', 'fi', 'gsw', 'se'), 
        7: ('mk', 'sr', 'uk', 'be', 'tg', 'ce', 'av', 'udm'), 
        8: ('el', 'grc', 'yi', 'pl', 'tr', 'os', 'ro', 'hu'), 
        9: ('ml', 'ta', 'te', 'kn', 'bn', 'mr', 'pa', 'gu'), 
        10: ('ckb', 'ug', 'ps', 'sd', 'fa', 'ur', 'az', 'br'), 
        11: ('jv', 'su', 'ms', 'mg', 'ig', 'sl', 'ht', 'vec'), 
        12: ('sm', 'to', 'haw', 'mi', 'st', 'tet', 'lv', 'fy'), 
        13: ('yue', 'zh', 'ja', 'ne', 'gl', 'oc', 'eo', 'co'), 
        14: ('ee', 'ts', 'hmn', 'lb', 'rm', 'gd', 'om', 'mt'), 
        15: ('km', 'th', 'bo', 'tk', 'sa', 'pap', 'kbd', 'kl')
    },
    save_path = f"{_HOME_DIR}/data/source/16Groups-train",
    test_path = f"{_HOME_DIR}/data/source/16Groups-test",
    total_tokens=9000000000,
    sample_alpha=0.3,
    test_portion=0.01
):
    langs = read_csv(_MADLAD400_STAT_PATH).keys()

    tok_char_rate_dict = {}
    tok_line_rate_dict = {}
    for lang in tqdm(langs, desc="Count tokens ..."):
        data = load_from_disk(_MADLAD400_DATA_PATH.format(lang=lang))
        tok_char_rate, tok_line_rate = char_line_tok_rate(data, tok, num=min(1000, len(data)))
        tok_char_rate_dict[lang] = tok_char_rate
        tok_line_rate_dict[lang] = tok_line_rate
        print(f"{lang}\t{tok_char_rate:.4f}\t{tok_line_rate:.4f}")
    
    lang_size_dict = {}
    lang_all_line_dict = read_csv(_MADLAD400_STAT_PATH)
    for lang in lang_all_line_dict.keys():
        lang_size_dict[lang] = int(lang_all_line_dict[lang] * (tok_line_rate_dict[lang] / tok_char_rate_dict[lang]))
    
    merge_raw_data(
        lang_groups=lang_groups,
        lang_size_dict=lang_size_dict,
        tok_line_rate_dict=tok_line_rate_dict,
        save_path=save_path,
        test_path=test_path,
        total_tokens=total_tokens,
        is_culturax=False,
        sample_alpha=sample_alpha,
        test_portion=test_portion
    )

def merge_shuffle(
    src_path,
    tgt_path,
    seed=0,
):
    d = load_from_disk(src_path)
    da = concatenate_datasets([v for v in d.values()])
    da = da.shuffle(seed=seed).flatten_indices().train_test_split(0.001)
    da.save_to_disk(tgt_path)
    return da

if __name__ == '__main__':
    # generate corpus for dynamic MoE layer extension from CulturaX 
    generate_culturax(
        tok = AutoTokenizer.from_pretrained("bigscience/bloom-1b7"),
        lang_groups = {
            0: ('ar', 'bn', 'hi', 'id', 'ta', 'ur'),
            1: ('de', 'fr', 'it', 'nl', 'vi', 'zh'),
            2: ('ja', 'ko', 'ru', 'te', 'th', 'uk'),
        },
        save_path = f"{_HOME_DIR}/data/source/3Groups-train",
        test_path = f"{_HOME_DIR}/data/source/3Groups-test",
        total_tokens=100000,
        sample_alpha=0.3,
        test_portion=0.01
    )

    # continue train MoE
    merge_shuffle(
        src_path = f"{_HOME_DIR}/data/source/3Groups-train",
        tgt_path = f"{_HOME_DIR}/data/source/3Groups-merge-train",
        seed=0
    )
