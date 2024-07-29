#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=harness
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:2

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="continual_training/OLMo_1B_source/step738020-unsharded-hf"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size 32 \
    --output_path eval_results