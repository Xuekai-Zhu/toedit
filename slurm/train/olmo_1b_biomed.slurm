#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=olmo_pretraining
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=continual_training/OLMo-1B-biomed/train.out
#SBATCH --error=continual_training/OLMo-1B-biomed/train.err
#SBATCH --gres=gpu:4
#SBATCH --qos=high

torchrun --nproc_per_node=4 --master_port=29216 OLMo/scripts/train.py continual_pretraining_config/OLMo-1B_biomed.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --reset_optimizer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
