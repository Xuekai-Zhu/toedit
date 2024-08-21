
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen2-1.5B \
  --repo-type model \
  --resume-download \
  --local-dir pre_trained_model/Qwen/Qwen2-1.5B \
  --local-dir-use-symlinks False
