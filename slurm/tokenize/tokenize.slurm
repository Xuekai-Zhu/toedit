#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=tokenize
#SBATCH --partition=RTX3090     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量    
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mem=100G



python OLMo/scripts/prepare_memmap_dataset.py data/bio/biomed_delete_greater_than_0.999_token/*.json.gz \
    -o data/bio/biomed_delete_greater_than_0.999_token_tokenized \
    --tokenizer OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
    --workers 4