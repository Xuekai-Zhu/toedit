#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=olma_pretraining
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=continual_training/1B_biomed_a100/train.out
#SBATCH --error=continual_training/1B_biomed_a100/train.err
#SBATCH --gres=gpu:2
#SBATCH --qos=high

torchrun --nproc_per_node=2 --master_port=29218 OLMo/scripts/train.py continual_pretraining_config/OLMo-1B.yaml \
    --save_overwrite \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
