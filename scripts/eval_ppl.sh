# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

export CURR_GPU=0
export GPUNUM=1
export MAX_GPU_ID=7
export TOTAL_GPU=8

NUM_BLOCK=100
SEED=0

DATA="${MAIN_DIR}/data/source/3Groups-test"
# DATA="${MAIN_DIR}/data/source/NewAdd4Langs-test"
# DATA="${MAIN_DIR}/data/source/128langs-test"

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

eval_ppl () {
    python -u src/ppl.py -m ${MODEL} -d ${DATA} -n ${NUM_BLOCK} -s ${SEED} -o ${OUT_FILE} &
}

eval_ppls () {
    NUM_MODEL="${#MODELS[@]}"

    for((mi=0;mi<${NUM_MODEL};mi++));  
    do
        set_gpu $CURR_GPU $GPUNUM
        MODEL=${MODELS[$mi]}
        OUT_FILE=${OUT_FILES[$mi]}
        eval_ppl
        CURR_GPU=$(($CURR_GPU+$GPUNUM))
        if [[ $CURR_GPU -gt $MAX_GPU_ID ]]
        then
            echo "GPU is full then wait ... (CURR_GPU=$CURR_GPU)"
            CURR_GPU=$(($CURR_GPU-$TOTAL_GPU))
            wait
        fi
    done

    wait
}

eval_expert_ppl () {
    python -u src/ppl_hard_router.py -m ${MODEL} -d ${DATA} -l ${LANG} -n ${NUM_BLOCK} -s ${SEED} -o ${OUT_FILE} &
}

eval_expert_ppls () {
    NUM_MODEL="${#MODELS[@]}"

    for((mi=0;mi<${NUM_MODEL};mi++));  
    do
        set_gpu $CURR_GPU $GPUNUM
        MODEL=${MODELS[$mi]}
        OUT_FILE=${OUT_FILES[$mi]}
        eval_expert_ppl
        CURR_GPU=$(($CURR_GPU+$GPUNUM))
        if [[ $CURR_GPU -gt $MAX_GPU_ID ]]
        then
            echo "GPU is full then wait ... (CURR_GPU=$CURR_GPU)"
            CURR_GPU=$(($CURR_GPU-$TOTAL_GPU))
            wait
        fi
    done

    wait
}

eval_ppl_map () {
    python -u src/ppl.py -m ${MODEL} -d ${DATA} -n ${NUM_BLOCK} -s ${SEED} -o ${OUT_FILE} -e ${EXPERT_MAP} &
}

eval_ppls_map () {
    NUM_MODEL="${#MODELS[@]}"

    for((mi=0;mi<${NUM_MODEL};mi++));  
    do
        set_gpu $CURR_GPU $GPUNUM
        MODEL=${MODELS[$mi]}
        OUT_FILE=${OUT_FILES[$mi]}
        eval_ppl_map
        CURR_GPU=$(($CURR_GPU+$GPUNUM))
        if [[ $CURR_GPU -gt $MAX_GPU_ID ]]
        then
            echo "GPU is full then wait ... (CURR_GPU=$CURR_GPU)"
            CURR_GPU=$(($CURR_GPU-$TOTAL_GPU))
            wait
        fi
    done

    wait
}

# MODELS=("${MAIN_DIR}/data/Qwen2.5-0.5B" "${MAIN_DIR}/data/Qwen2.5-1.5B")
# OUT_FILES=("${MAIN_DIR}/log/Qwen2.5-0.5B.ppl.json" "${MAIN_DIR}/log/Qwen2.5-1.5B.ppl.json")

# eval_ppls

LANG="zh"
MODELS=("${MAIN_DIR}/log/BloomMoEDyn/560M-EPS0.6-EXP3-S2_SD8/checkpoint-1500")
OUT_FILES=("${MAIN_DIR}/log/BloomMoEDyn-560M-EPS0.6-EXP3-S1_SD8.expert-zh.ppl.json")

eval_expert_ppls

# MODELS=("${MAIN_DIR}/log/BloomMoEDyn/560M-EPS0.6-EXP3-S2_SD8/checkpoint-1500")
# OUT_FILES=("${MAIN_DIR}/log/BloomMoEDyn-560M-EPS0.6-EXP3-S1_SD8.ppl.json")
# EXPERT_MAP="${MAIN_DIR}/data/expert-map/3Groups-train.json"

# eval_ppls_map

