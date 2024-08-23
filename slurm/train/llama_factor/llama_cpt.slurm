#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=llama_factory
#SBATCH --partition=ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:4


unset all_proxy   
unset http_proxy       
unset https_proxy 
export NCCL_P2P_DISABLE=1

llamafactory-cli train /home/xkzhu/scaling_down_data/config/llama_factory/qwen2_cpt.yaml
