#!/bin/bash
#SBATCH -J olma_pretraining
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o bio_revised+orca_2.out
#SBATCH -e bio_revised+orca_2.err
#SBATCH --gres=gpu:4

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py310

torchrun --nproc_per_node=4 --master_port=29216 OLMo/scripts/train.py config/bio/biomed+orca/bigai_OLMo-1B_biomed_revised+orca.yaml \
    --save_overwrite \
    --load_path=/home/zhuxuekai/scratch2_nlp/scaling_down_data/continual_training/bio/OLMo-1B-biomed-revised+orca/latest
