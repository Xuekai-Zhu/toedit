#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=inference
#SBATCH --partition=ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=high

MODEL="/home/xkzhu/scaling_down_data/continual_training/OLMo-1B-biomed-0.001-v2/step3684-unsharded" 

# python /home/xkzhu/scaling_down_data/OLMo/hf_olmo/convert_olmo_to_hf_new.py \
#     --input_dir ${MODEL} \
#     --tokenizer_json_path /home/xkzhu/scaling_down_data/OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
#     --output_dir ${MODEL}-hf

unset all_proxy   
unset http_proxy       
unset https_proxy 

export HF_ENDPOINT=https://hf-mirror.com


DOMAIN='biomedicine'

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
bash scripts/inference.sh ${DOMAIN} ${MODEL}-hf ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}