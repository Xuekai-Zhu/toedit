#!/bin/bash
#SBATCH -J olma_pretraining
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o openweb_math.out
#SBATCH -e openweb_math.err
#SBATCH --gres=gpu:4

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate olmo

torchrun --nproc_per_node=4 --master_port=29216 OLMo/scripts/train.py config/math/OLMo-1B_openweb_math.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=pre_trained_model/step738020-unsharded
