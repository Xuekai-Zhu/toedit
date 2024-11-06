#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=harness
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:4
#SBATCH --mem=40G

unset all_proxy   
unset http_proxy       
unset https_proxy 

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="/data1/xkzhu/pre_trained_model/meta-llama/Meta-Llama-3-8B"

# python OLMo/scripts/unshard.py ${MODEL} ${MODEL}-unshard

# python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
#     --input_dir ${MODEL} \
#     --tokenizer_json_path OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
#     --output_dir ${MODEL}-hf

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks hellaswag,arc_easy,arc_challenge,social_iqa,winogrande,piqa,boolq,openbookqa \
    --batch_size auto:4 \
    --output_path eval_results/general/

