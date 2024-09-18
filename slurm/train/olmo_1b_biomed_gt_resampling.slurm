#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=gt_re
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=biomed-gt_resampling.out
#SBATCH --error=biomed-gt_resampling.err
#SBATCH --gres=gpu:4

torchrun --nproc_per_node=4 --master_port=29219 OLMo/scripts/train.py config/bio/OLMo-1B_biomed_gt_resampling.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
