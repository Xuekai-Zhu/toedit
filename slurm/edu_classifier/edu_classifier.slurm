#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=scores
#SBATCH --partition=RTX4090  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:2
#SBATCH --qos=high

deepspeed edu_classifier.py \
    --model_name_or_path /data1/xkzhu/pre_trained_model/HuggingFaceFW/fineweb-edu-classifier \
    --test_file data/cosmopedia/cosmopedoa_all_jsonl_index \
    --per_device_eval_batch_size 36 \
    --max_length 512 \
    --output_dir edu_scores/fineweb-edu-classifier/cosmopedia_all

