#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=pre_training
#SBATCH --partition=RTX4090      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:4        
#SBATCH --output=%j.out
#SBATCH --error=%j.err

torchrun --nproc_per_node=4 OLMo/scripts/train.py OLMo/configs/my_config/test_OLMo.yaml \
    --save_overwrite
