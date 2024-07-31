#!/bin/bash

#SBATCH --account=xkzhu
#SBATCH --job-name=convert2hf
#SBATCH --partition=RTX3090      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G

MODEL="continual_training/OLMo-1B-Arithmo-Data/step457-unsharded" 

# python OLMo/scripts/unshard.py ${MODEL} ${MODEL}-unshard

python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
    --input_dir ${MODEL} \
    --tokenizer_json_path OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --output_dir ${MODEL}-hf