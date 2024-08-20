#!/bin/bash
#SBATCH -J olma_pretraining
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o 1b_orca.out
#SBATCH -e 1b_orca.err
#SBATCH --gres=gpu:2

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py39

llamafactory-cli train config/llama_factory/llama3_full_sft_ds3.yaml
