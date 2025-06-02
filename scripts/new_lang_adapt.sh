export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUNUM=8

# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

export MASTER_PORT=16899

# export WANDB_MODE=online
export WANDB_MODE=offline

STAGE=1
export TASK_NAME="2B-Add4Langs-Mapping"

# export MODEL="BloomMoEDyn"
export MODEL="GemMoEDyn"

# MODEL_NAME="${MAIN_DIR}/data/BloomMoEDyn/560M-EPS0.6-EXP9-ADD2"
MODEL_NAME="${MAIN_DIR}/data/GemMoEDyn/2B-EPS0.6-EXP9-ADD2"

export RESUME=False

export TRAIN_BS=4
export EVAL_BS=1

export DATASET_PATH="${MAIN_DIR}/data/datasets/NewAdd4Langs-train-Gemma"

DATASET_MAP_PATH="${MAIN_DIR}/data/expert-map/new4langs_dataset_map.json"

# export CONFIG_FILE="${MAIN_DIR}/data/Deepspeed-Configs/zero2.yaml"
export CONFIG_FILE="${MAIN_DIR}/data/Deepspeed-Configs/zero3.yaml"

export GRADIENT_ACC=30

CONSECUTIVE_NUM=64

export BLOCK_SIZE=2048

export SEED=8

export LR=2e-4
export NUM_STEPS=100
export NUM_SAVE_STEPS=100
export EVAL_STEP=5000
export NUM_WORKERS=0
export LOGGING_STEPS=1

ROUTER_COEF_S1=1.28

export TRAIN_IDX=-1

export RESUME=False

export ADD_PARAMETERS=""

PREFIX="${MODEL}/${TASK_NAME}_SD${SEED}"

if [ "${RESUME}" != "False" ];
then
PREFIX="${PREFIX}_resume"
ADD_PARAMETERS="${ADD_PARAMETERS} --resume_from_checkpoint ${RESUME}"
fi

MODEL_DIR="log/$PREFIX"
LOG_FILE="log/${PREFIX}.log"

mkdir -p $MODEL_DIR


accelerate launch \
    --config_file ${CONFIG_FILE} \
    --num_processes ${GPUNUM} \
    --main_process_port ${MASTER_PORT} \
    --num_machines 1 model/fsdp_train.py \
    --model_name ${MODEL_NAME} \
    --tokenizer_path ${MODEL_NAME} \
    --dataset_name ${DATASET_PATH} \
    --max_seq_length ${BLOCK_SIZE} \
    --max_steps ${NUM_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${NUM_SAVE_STEPS} \
    --num_workers ${NUM_WORKERS} \
    --bf16 True \
    --packing True \
    --seed ${SEED} \
    --output_dir ${MODEL_DIR} \
    --per_device_train_batch_size ${TRAIN_BS} \
    --train_layer_idx ${TRAIN_IDX} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    ${ADD_PARAMETERS} \
    --use_flash_attn \
    --map_to_expert True \
    --only_tune_moe True \
    --eval_data_ratio 0.01 \
    --router_aux_loss_coef ${ROUTER_COEF_S1} \
    --dataset_key2id ${DATASET_MAP_PATH} \
    --consecutive_num ${CONSECUTIVE_NUM} \
    --warmup_ratio 0.03 2>&1 >$LOG_FILE