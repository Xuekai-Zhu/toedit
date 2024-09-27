#!/bin/bash
#SBATCH -J dolma
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o dolma_revised_3.out
#SBATCH -e dolma_revised_3.err
#SBATCH --gres=gpu:4

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py310

torchrun --nproc_per_node=4 --master_port=29210 OLMo/scripts/train.py config/dolma/OLMo-1B_dolma_bigai.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=continual_training/dolma/OLMo-1B-dolma/step10000-unsharded