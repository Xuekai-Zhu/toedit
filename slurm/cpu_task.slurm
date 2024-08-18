#!/bin/bash

#SBATCH --account=xkzhu
#SBATCH --job-name=merge_data
#SBATCH --partition=RTX3090      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=4  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G

python decode_tokens.py   

