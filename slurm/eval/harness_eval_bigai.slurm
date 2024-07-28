#SBATCH -J olma_pretraining
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="continual_training/OLMo-1B-biomed+orca/step4000-unshard-hf"

lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --device cuda:0 \
    --batch_size 16 \
    --output_path eval_results