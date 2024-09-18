#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=tokenize
#SBATCH --partition=RTX4090     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量    
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G

python prepare_ai_mo_data.py data/AI-MO-tokenized \
    --dataset_name data/AI-MO \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json