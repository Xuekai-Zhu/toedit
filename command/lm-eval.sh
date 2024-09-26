CUDA_VISIBLE_DEVICES="3"
MODEL="continual_training/dolma/OLMo-1B-dolma/step10000-unsharded-hf"

accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL},trust_remote_code=True \
    --tasks hellaswag,arc_challenge,social_iqa,winogrande,piqa \
    --batch_size 16 \
    --output_path eval_results/general/