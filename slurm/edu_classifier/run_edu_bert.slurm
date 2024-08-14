#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=edu_scores
#SBATCH --partition=RTX4090  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1


python baselines/run_edu_bert.py \
    --model_name="/data1/xkzhu/pre_trained_model/HuggingFaceFW/fineweb-edu-classifier" \
    --dataset_name="data/AI-MO/NuminaMath-CoT" \
    --output_dir="baselines/edu_scores"