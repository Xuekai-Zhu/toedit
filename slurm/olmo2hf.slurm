#!/bin/bash

#SBATCH --account=xkzhu
#SBATCH --job-name=convert2hf
#SBATCH --partition=RTX3090      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
    --input_dir continual_training/1B_biomed_a100/step738020-unsharded \
    --tokenizer_json_path OLMo/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
    --output_dir continual_training/1B_biomed_a100/step738020-unsharded-hf