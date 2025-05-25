#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=pre_training
#SBATCH --partition=RTX4090      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python check_data.py