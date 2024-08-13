
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download AI-MO/NuminaMath-CoT \
  --repo-type dataset \
  --resume-download \
  --local-dir data/AI-MO/NuminaMath-CoT \
  --local-dir-use-symlinks False
