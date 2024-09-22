#!/bin/bash
#SBATCH -J infer
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py310


SOURCE_PATH="/home/zhuxuekai/scratch2_nlp/scaling_down_data"
MODEL=${SOURCE_PATH}/continual_training/finance/OLMo-1B-finance-orca/step5740-unsharded

# python $SOURCE_PATH/OLMo/hf_olmo/convert_olmo_to_hf_new.py \
#     --input_dir ${MODEL} \
#     --tokenizer_json_path ${SOURCE_PATH}/OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
#     --output_dir ${MODEL}-hf

export HF_ENDPOINT=https://hf-mirror.com
DOMAIN='finance'

# if the model can fit on a single GPU: set MODEL_PARALLEL=False
# elif the model is too large to fit on a single GPU: set MODEL_PARALLEL=True
MODEL_PARALLEL=False

# number of GPUs, chosen from [1,2,4,8]
N_GPU=1

# # AdaptLLM-7B pre-trained from Llama1-7B
# add_bos_token=False # this is set to False for AdaptLLM, and True for instruction-pretrain
# bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/medicine-LLM' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# # AdaptLLM-13B pre-trained from Llama1-13B
# add_bos_token=False
# bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/medicine-LLM-13B' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# medicine-Llama-8B pre-trained from Llama3-8B in Instruction Pretrain
add_bos_token=False
bash scripts/inference_bigai.sh ${DOMAIN} ${MODEL}-hf ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}