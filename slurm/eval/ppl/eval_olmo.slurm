#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=eval_olmo
#SBATCH --partition=ADA6000  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=high

python eval_ppl_for_olmo_ds.py \
    --model_name_or_path /data1/xkzhu/probability/sclaing_law/pre_training_results/GPT-2/dolma/epoch_5/epoch_4_steps_160280 \
    --per_device_eval_batch_size 64 \
    --max_length 1024 \
    --output_dir eval_results/ppl/dolma