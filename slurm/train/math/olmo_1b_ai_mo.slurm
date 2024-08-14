#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=AI-MO-Edu
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=AI-MO-Edu.out
#SBATCH --error=AI-MO-Edu.err
#SBATCH --gres=gpu:4
#SBATCH --qos=high

torchrun --nproc_per_node=4 --master_port=29201 OLMo/scripts/train.py config/fine-tuning/OLMo-1B.yaml \
    --save_overwrite \
    --data.paths=[data/AI-MO/Edu_filter_Num_CoT-tokenized/input_ids.npy] \
    --data.label_mask_paths=[data/AI-MO/Edu_filter_Num_CoT-tokenized/label_mask.npy] \
    --data.drop_last=true \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
