#!/bin/bash
#SBATCH -J harness
#SBATCH -p HGX
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1

module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.163  

source /home/zhuxuekai/miniconda3/bin/activate lm-eval


MODEL="/home/zhuxuekai/scratch2_nlp/scaling_law4AI_data/pre_training_results/olma/1ep/2_layer_dolma/latest-unsharded-hf"

# python OLMo/hf_olmo/convert_olmo_to_hf_new.py \
#      --input_dir ${MODEL} \
#      --tokenizer_json_path OLMo/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json \
#      --output_dir ${MODEL}-hf

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks truthfulqa,logiqa2,winogrande,piqa,arc_easy,boolq,openbookqa \
    --batch_size auto:4 \
    --output_path /home/zhuxuekai/scratch2_nlp/scaling_down_data/ICLR_rebuttal/olma-results/ \
    --trust_remote_code
