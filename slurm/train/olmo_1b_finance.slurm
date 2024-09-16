#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=finance
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=finance.out
#SBATCH --error=finance.err
#SBATCH --gres=gpu:4
#SBATCH --qos=high

torchrun --nproc_per_node=4 --master_port=29221 OLMo/scripts/train.py config/finance/OLMo-1B_finance.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
