#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=harness
#SBATCH --partition=ADA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=high

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="fine-tuning/OLMo-1B-AI-MO-Edu/step4250-unsharded"

python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
    --input_dir ${MODEL} \
    --tokenizer_json_path OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --output_dir ${MODEL}-hf

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL}-hf,trust_remote_code=True \
    --tasks hendrycks_math,gsm8k,gpqa,arc_challenge,mmlu \
    --batch_size 32 \
    --output_path eval_results/OLMo-1B-AI-MO-Edu-step4250-unsharded-hf