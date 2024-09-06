#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=online_sampling
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=biomed-0.9_beta_1.5.out
#SBATCH --error=biomed-0.9_beta_1.5.err
#SBATCH --gres=gpu:4

torchrun --nproc_per_node=4 --master_port=29219 OLMo/scripts/train.py config/bio/OLMo-1B_biomed_0.9_resmapling_llama_beta_1.5.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/
