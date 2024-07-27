#!/bin/bash
#SBATCH -J hf
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o 1b_orca.out
#SBATCH -e 1b_orca.err
#SBATCH --gres=gpu:1

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate olmo

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
bash scripts/inference.sh ${DOMAIN} '/home/zhuxuekai/scratch2_nlp/scaling_down_data/continual_training/OLMo-1B-orca-lr-1.0e-6/step3000-unshard-hf' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}