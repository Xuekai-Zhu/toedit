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

huggingface-cli download HuggingFaceFW/fineweb-edu \
  --repo-type dataset \
  --resume-download \
  --local-dir path/to/your/dir \
  --local-dir-use-symlinks False

huggingface-cli download Muennighoff/natural-instructions \
  --repo-type dataset \
  --force-download \
  --resume-download False \
  --local-dir data/Muennighoff/natural-instructions \
  --local-dir-use-symlinks False