#!/bin/bash
#SBATCH -J bio_orca
#SBATCH -p HGX
#SBATCH --qos=lv2
#SBATCH --time=2-00:00:00
#SBATCH -o bio_orca.out
#SBATCH -e bio_orca.err
#SBATCH --gres=gpu:4
#SBATCH --account=research 


module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate llama-factory

llamafactory-cli train slurm/yaml/llama3_lora_bio_orca.yaml
