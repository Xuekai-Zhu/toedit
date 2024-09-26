#!/bin/bash
#SBATCH -J llama3_lora_finance_orca
#SBATCH -p HGX
#SBATCH --qos=lv2
#SBATCH --time=2-00:00:00
#SBATCH -o lora_finance_orca_Llama-3-8B.out
#SBATCH -e lora_finance_orca_Llama-3-8B.err
#SBATCH --gres=gpu:4
#SBATCH --account=research 


module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate llama-factory

llamafactory-cli train llama3_lora_finance_orca.yaml
