#SBATCH -J olma_pretraining
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:2

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="/home/zhuxuekai/scratch2_nlp/scaling_down_data/continual_training/OLMo-1B-open-web-math/step4187-unsharded"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size 16 \
    --output_path eval_results