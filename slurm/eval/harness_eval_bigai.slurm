#!/bin/bash

#SBATCH -J olma_pretraining
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --gres=gpu:1

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate olmo

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="continual_training/OLMo-1B-open-web-math/step4187-unsharded"
HF_MODEL="continual_training/OLMo-1B-open-web-math/step4187-unsharded-hf"

python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
    --input_dir ${MODEL} \
    --tokenizer_json_path OLMo/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
    --output_dir ${HF_MODEL}


accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${HF_MODEL},trust_remote_code=True \
    --tasks leaderboard_math_hard \
    --batch_size 32 \
    --output_path eval_results