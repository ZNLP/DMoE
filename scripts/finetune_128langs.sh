# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

langs=("af" "am" "ar" "as" "av" "az" "ba" "be" "bg" "bn" "bo" "br" "bs" "ca" "ce" "ceb" "ckb" "cnh" "co" "cs" "cv" "cy" "da" "de" "dv" "ee" "el" "en" "eo" "es" "et" "eu" "fa" "fi" "fil" "fo" "fr" "fy" "ga" "gd" "gl" "grc" "gsw" "gu" "ha" "haw" "he" "hi" "hil" "hmn" "hr" "ht" "hu" "hy" "id" "ig" "ilo" "is" "it" "ja" "jv" "ka" "kaa" "kbd" "kha" "kk" "kl" "km" "kn" "ko" "ku" "ky" "la" "lb" "lg" "lo" "lt" "lus" "lv" "mg" "mi" "mk" "ml" "mn" "mr" "ms" "mt" "my" "ne" "nl" "no" "ny" "oc" "om" "or" "os" "pa" "pap" "pl" "ps" "pt" "rm" "ro" "ru" "rw" "sa" "sah" "sd" "se" "si" "sk" "sl" "sm" "sn" "so" "sq" "sr" "st" "su" "sv" "sw" "ta" "te" "tet" "tg" "th" "ti" "tk" "to" "tr" "ts" "tt" "tyv" "udm" "ug" "uk" "ur" "uz" "vec" "vi" "xh" "yi" "yo" "yue" "zh" "zu")

# run with bloom-1b7 for example
MODEL_PATH="bigscience/bloom-1b7"
DATA_PATH="./data/MADLAD-400-for-Cluster"
# the save path for last 3 FFN layer parameters
OUTPUT_DIR="./data/multi-128-10steps-1.7b"
# the save path for the delta parameter of each layer
LAYER_DIR="./img/multi-128-10steps-1.7b/delta-layer-data"

SCRIPT="./src/finetune_delta_128langs.py"

GPUNUM=1
CURR_GPU=0

set_gpu () {
    SID=$1
    SET_GPU_NUM=$2
    GPUS="${SID}"
    for((g=1;g<${SET_GPU_NUM};g++));
    do
        GPUS="${GPUS},$(($g+$SID))"
    done
    export CUDA_VISIBLE_DEVICES=$GPUS
}

LOG_FILE="./log/multi-128-10steps/1.7b/train128$lang.log"

mkdir -p "$(dirname "$LOG_FILE")"
> "$LOG_FILE"

for lang in "${langs[@]}"; do
    python -u ./src/process_128lang_finetune_data.py ${lang}

    LANG_DATA_PATH="$DATA_PATH/$lang"
    LANG_OUTPUT_DIR="$OUTPUT_DIR/$lang"

    LOG_FILE="./log/multi-128-10steps/1.7b/train128_$lang.log"

    echo "Running training for language: $lang" | tee -a "$LOG_FILE"
    set_gpu $CURR_GPU $GPUNUM
    
    python -u $SCRIPT "$lang" "$MODEL_PATH" "$LANG_DATA_PATH" "$LANG_OUTPUT_DIR" "$LAYER_DIR" 2>&1 >> "$LOG_FILE" &
    
    CURR_GPU=$(($CURR_GPU+$GPUNUM))
    if [[ $CURR_GPU -gt 7 ]]
    then
        echo "GPU is full then wait ... (CURR_GPU=$CURR_GPU)"
        CURR_GPU=$(($CURR_GPU-8))
        wait
    fi
done

wait