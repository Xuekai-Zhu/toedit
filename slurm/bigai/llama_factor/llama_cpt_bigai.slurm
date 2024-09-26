#!/bin/bash
#SBATCH -J llama_factory
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o llama_factory.out
#SBATCH -e llama_factory.err
#SBATCH --gres=gpu:2

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate llama-factory

llamafactory-cli train llama_factory_config/llama3_lora_pretrain.yaml
