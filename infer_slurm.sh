#!/bin/bash
#SBATCH --job-name=qw-7b
#SBATCH ... your slurm options here ...


conda activate your_env_name

################ Config ################
# Declare associative array properly
declare -A MODEL_DICT=( 
    ["Llama-3-8B"]="meta-llama/Meta-Llama-3-8B"
    ["Llama-3-70B"]="meta-llama/Meta-Llama-3-70B" 
    ["Qwen2.5-0.5B"]="Qwen/Qwen2.5-0.5B-instruct"
    ["Qwen2-7B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ["Qwen2-14B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    ["Qwen2-32B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)
# Llama-3-8B or Qwen2.5-0.5B Qwen2.5-7B
MODEL_SERIES="Qwen2-7B"
MODEL_TYPE="qwen" # base or instruct or qwen
INFER_MODE="zs" # zs or sft
# pew_c3_ for pew
DATA_PATH="dataset/sft_wvs_"  # "dataset/sft_wvs_debug_" for debug
CKPT_PATH="Llama-3-8B_instruct_epoch_5/"
VAL_BATCH_SIZE=8
################ End Config ################

if [[ ${MODEL_TYPE} == "base" ]]; then
    MODEL_NAME="${MODEL_DICT[${MODEL_SERIES}]}"
elif [[ ${MODEL_TYPE} == "instruct" ]]; then
    MODEL_NAME="${MODEL_DICT[${MODEL_SERIES}]}-Instruct"
elif [[ ${MODEL_TYPE} == "qwen" ]]; then
    MODEL_NAME="${MODEL_DICT[${MODEL_SERIES}]}"
fi

# Save directory
SAVE_DIR="output/Replace_${MODEL_SERIES}_${MODEL_TYPE}_${INFER_MODE}"

echo "Model Name: ${MODEL_SERIES}_${MODEL_TYPE}"
echo "Save directory: $SAVE_DIR"


if [[ ${INFER_MODE} == "zs" ]]; then
    python inference.py \
    --model_name ${MODEL_NAME}  \
    --model_type ${MODEL_TYPE} \
    --data_path ${DATA_PATH} \
    --use_auditnlg \
    --dataset custom_dataset \
    --batching_strategy "padding" \
    --val_batch_size $VAL_BATCH_SIZE \
    --max_sent_len 256 \
    --save_dir ${SAVE_DIR}
elif [[ ${INFER_MODE} == "sft" ]]; then
    python inference.py \
    --model_name ${MODEL_NAME}  \
    --model_type ${MODEL_TYPE} \
    --data_path ${DATA_PATH} \
    --use_auditnlg \
    --dataset custom_dataset \
    --batching_strategy "padding" \
    --val_batch_size $VAL_BATCH_SIZE \
    --save_dir ${SAVE_DIR} \
    --max_sent_len 196 \
    --peft_model SimLLMCultureDist/output/${CKPT_PATH}
fi