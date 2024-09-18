#!/bin/bash
#SBATCH -J olma_pretraining
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o finance.out
#SBATCH -e finance.err
#SBATCH --gres=gpu:4

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate py310

torchrun --nproc_per_node=4 --master_port=29216 OLMo/scripts/train.py config/finance/OLMo-1B_finance_bigai_llama_revised.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
