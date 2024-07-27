#!/bin/bash
#SBATCH -J hf
#SBATCH -p DGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o 1b_orca.out
#SBATCH -e 1b_orca.err
#SBATCH --gres=gpu:1

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate olmo

MODEL="/home/zhuxuekai/scratch2_nlp/scaling_down_data/continual_training/OLMo-1B-orca-lr-1.0e-6/step3000" 

python OLMo/scripts/unshard.py ${MODEL} ${MODEL}-unshard

python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
    --input_dir ${MODEL}-unshard \
    --tokenizer_json_path OLMo/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json \
    --output_dir ${MODEL}-unshard-hf
