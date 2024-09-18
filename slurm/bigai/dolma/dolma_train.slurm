#!/bin/bash
#SBATCH -J dolma
#SBATCH -p HGX
#SBATCH --qos=lv2
#SBATCH --time=2-00:00:00
#SBATCH -o dolma_revised.out
#SBATCH -e dolma_revised.err
#SBATCH --gres=gpu:4
#SBATCH --account=research 

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py310

torchrun --nproc_per_node=4 --master_port=29216 OLMo/scripts/train.py config/dolma/OLMo-1B_dolma_revised.yaml \
    --save_overwrite 