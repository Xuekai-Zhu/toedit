#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=down_load
#SBATCH --partition=ADA6000     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量    
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G


python OLMo/scripts/unshard.py continual_training/dolma/OLMo-1B-dolma-revised/step15000 continual_training/dolma/OLMo-1B-dolma-revised/step15000-unshard