#!/bin/bash
#SBATCH --job-name=qw-7b
#SBATCH ... your slurm options here ...


conda activate your_env_name

# Declare associative array properly
declare -A MODEL_DICT=( 
    ["Llama-3-8B"]="meta-llama/Meta-Llama-3-8B" 
    ["Qwen2-7B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ["Qwen2-14B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    ["Qwen2-14B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

# Llama-3-8B or Qwen2.5-0.5B Qwen2.5-7B Baichuan2-7B GLM-4
MODEL_SERIES="Qwen2-7B"
MODEL_TYPE="qwen" # base or instruct or qwen or GLM-4 baichuan
EPOCH=5
DATA_PATH="dataset/sft_wvs_"  # "dataset/sft_wvs_debug_" for debug

if [[ ${MODEL_TYPE} == "base" ]]; then
    MODEL_NAME="${MODEL_DICT[${MODEL_SERIES}]}"
elif [[ ${MODEL_TYPE} == "instruct" ]]; then
    MODEL_NAME="${MODEL_DICT[${MODEL_SERIES}]}-Instruct"
elif [[ ${MODEL_TYPE} == "qwen" ]]; then
    MODEL_NAME="${MODEL_DICT[${MODEL_SERIES}]}"
fi

# Save directory
SAVE_DIR="output/${MODEL_SERIES}_${MODEL_TYPE}_epoch_${EPOCH}"

echo "Model Name: $MODEL_NAME"
echo "Save directory: $SAVE_DIR"

python finetuning.py  \
    --project "first_token_alignment" \
    --job_type "train" \
    --name ${MODEL_SERIES}_${MODEL_TYPE}_epoch_${EPOCH} \
    --use_peft --peft_method lora \
    --quantization  --dataset custom_dataset \
    --model_type ${MODEL_TYPE} \
    --data_path ${DATA_PATH} \
    --model_name ${MODEL_NAME}  \
    --output_dir SimLLMCultureDist/${SAVE_DIR} \
    --batch_size_training 8 \
    --batching_strategy "padding" \
    --num_epochs ${EPOCH} \
    --max_sent_len 256 \
    --use_wandb