#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=llama_factory
#SBATCH --partition=ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:2


llamafactory-cli train config/llama_factory/qwen2_cpt_bigai.yaml
