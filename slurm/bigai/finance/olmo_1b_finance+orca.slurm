#!/bin/bash
#SBATCH -J inference
#SBATCH -p DGX
#SBATCH --qos=lv2
#SBATCH --time=2-00:00:00
#SBATCH -o finance+orca.out
#SBATCH -e finance+orca.err
#SBATCH --gres=gpu:4
#SBATCH --account=research 

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py310

torchrun --nproc_per_node=4 --master_port=29211 OLMo/scripts/train.py config/finance/OLMo-1B_finance+orca.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
