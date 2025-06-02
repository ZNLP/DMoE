# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

MODEL_TYPE=bloom
MODEL_SIZE=560m
MODEL_PATHS="bigscience/bloom-560m bigscience/bloom-560m bigscience/bloom-560m"
INDEX=15728671
OUT_PATH="./data/BloomMoEDyn/4x560M-IDX15728671"

python src/merge2moedyn.py \
    -m ${MODEL_TYPE} \
    -s ${MODEL_SIZE} \
    -p ${MODEL_PATHS} \
    -i ${INDEX} \
    -o ${OUT_PATH}
