#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=olmo_pretraining
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:4

torchrun --nproc_per_node=4 --master_port=29216 OLMo/scripts/train.py config/bio/biomed+orca/OLMo-1B_biomed_context+orca__llama_8B_revised.yaml \
    --save_overwrite \
    --reset_trainer_state \
    --load_path=pretrained_model/step738020-unsharded
