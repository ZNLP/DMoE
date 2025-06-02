export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUNUM=8

# Modify this to your local path, e.g., /home/ubuntu/DMoE
export MAIN_DIR="path2DMoE"
cd ${MAIN_DIR}

export MASTER_PORT=16899

# export WANDB_MODE=online
export WANDB_MODE=offline

NUM_EXPERT=3
STAGE=1
export TASK_NAME="560M-EPS0.6-EXP${NUM_EXPERT}-S$STAGE"

export MODEL="BloomMoEDyn"

# eps=0.8
# MODEL_NAME="${MAIN_DIR}/data/${MODEL}/${NUM_EXPERT}x560M-IDX14680064"
# eps=0.6
MODEL_NAME="${MAIN_DIR}/data/${MODEL}/${NUM_EXPERT}x560M-IDX15728671"

export RESUME=False

export TRAIN_BS=4
export EVAL_BS=1

export DATASET_PATH="${MAIN_DIR}/data/datasets/${NUM_EXPERT}Groups-train"

# export CONFIG_FILE="${MAIN_DIR}/data/Deepspeed-Configs/zero2.yaml"
export CONFIG_FILE="${MAIN_DIR}/data/Deepspeed-Configs/zero3.yaml"

export GRADIENT_ACC=30

CONSECUTIVE_NUM=64

export BLOCK_SIZE=2048

export SEED=8

export LR=1e-4
export NUM_STEPS=150
export NUM_SAVE_STEPS=150
export EVAL_STEP=5000
export NUM_WORKERS=0
export LOGGING_STEPS=1

export ROUTER_COEF_S1=1.28

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
    --num_machines 1 src/dmoe_train.py \
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
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    ${ADD_PARAMETERS} \
    --map_to_expert True \
    --eval_data_ratio 0.01 \
    --router_aux_loss_coef ${ROUTER_COEF_S1} \
    --consecutive_num ${CONSECUTIVE_NUM} \
    --warmup_ratio 0.03 2>&1 >$LOG_FILE


### Stage 2
STAGE=2
export TASK_NAME="560M-EPS0.6-EXP${NUM_EXPERT}-S$STAGE"

MODEL_NAME="./$MODEL_DIR/checkpoint-$NUM_STEPS"
export DATASET_PATH="${MAIN_DIR}/data/datasets/${NUM_EXPERT}Groups-merge-train"

LR=1e-4
NUM_STEPS=1500
NUM_SAVE_STEPS=1500
ROUTER_COEF_S2=0.02

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
    --num_machines 1 src/dmoe_train.py \
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
    --use_gradient_checkpointing \
    --learning_rate ${LR}  \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    ${ADD_PARAMETERS} \
    --router_aux_loss_coef ${ROUTER_COEF_S2} \
    --consecutive_num ${CONSECUTIVE_NUM} \
    --warmup_ratio 0.03 2>&1 >$LOG_FILE
