#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=down_load
#SBATCH --partition=RTX4090     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量    
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G


python OLMo/scripts/unshard.py continual_training/1B_biomed_a100/step0 continual_training/1B_biomed_a100/step0-unshard