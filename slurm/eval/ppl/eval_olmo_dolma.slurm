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

source /home/zhuxuekai/miniconda3/bin/activate pt310

MODEL="/home/zhuxuekai/scratch2_nlp/scaling_law4AI_data/pre_training_results/GPT-2/dolma/epoch_5/epoch_4_steps_160280"
OUTPUT="eval_results/ppl/dolma"

python eval_ppl_for_olmo_ds.py \
    --model_name_or_path ${MODEL} \
    --per_device_eval_batch_size 64 \
    --max_length 1024 \
    --output_dir ${OUTPUT}