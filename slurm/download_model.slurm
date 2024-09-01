#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=down_load
#SBATCH --partition=RTX3090     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量    
#SBATCH --output=%j.out
#SBATCH --error=%j.err

unset all_proxy   
unset http_proxy       
unset https_proxy 

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download meta-llama/Meta-Llama-3.1-8B \
  --repo-type model \
  --resume-download \
  --local-dir pre_trained_model/meta-llama/Meta-Llama-3.1-8B \
  --local-dir-use-symlinks False