#!/bin/bash
#SBATCH -J llama_factory
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o llama_factory.out
#SBATCH -e llama_factory.err
#SBATCH --gres=gpu:2


llamafactory-cli train llama_factory_config/llama3_lora_pretrain.yaml
