# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

langs=("ar" "bn" "de" "fr" "hi" "id" "it" "ja" "ko" "nl" "ru" "ta" "te" "th" "uk" "ur" "vi" "zh")

# run with bloom-1b7 for example
MODEL_PATH="bigscience/bloom-1b7"
MODEL_NAME="bloom-1.7b"
NUM_GROUP=9
LOG_FILE="./log/multi-18-10steps/bloom_1d7_18.log"

SCRIPT="./src/finetune_delta_18langs.py"

mkdir -p "$(dirname "$LOG_FILE")"

> "$LOG_FILE"

GPU=0

counter=0

for lang in "${langs[@]}"; do
    echo "Running training for language: $lang" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU python -u $SCRIPT "$lang" ${MODEL_PATH} ${MODEL_NAME} 2>&1 | tee -a "$LOG_FILE"
    wait
done


# Stage-2 calculate delta and language similarity
python -u ./src/delta_smi/bloom.py ${MAIN_DIR} ${MODEL_NAME} | tee -a "$LOG_FILE"
# python -u ./src/delta_smi/gemma.py ${MAIN_DIR} | tee -a "$LOG_FILE"

# Stage-3 language clustering
python -u ./src/balanced_cluster_18langs.py ${MODEL_NAME} ${NUM_GROUP} | tee -a "$LOG_FILE"