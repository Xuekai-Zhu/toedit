#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=open-web-math
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=openweb_math_2b.out
#SBATCH --error=openweb_math_2b.err
#SBATCH --gres=gpu:4

torchrun --nproc_per_node=4 --master_port=29201 OLMo/scripts/train.py config/math/OLMo-1B_openweb_math_2b.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
