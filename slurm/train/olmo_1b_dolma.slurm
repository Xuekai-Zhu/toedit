#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=dolma
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=dolma.out
#SBATCH --error=dolma.err
#SBATCH --gres=gpu:4

torchrun --nproc_per_node=4 --master_port=29221 OLMo/scripts/train.py config/dolma/OLMo-1B_dolma.yaml \
    --save_overwrite 
