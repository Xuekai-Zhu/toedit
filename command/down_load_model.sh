
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download open-web-math/open-web-math \
  --repo-type dataset \
  --resume-download \
  --local-dir data/open-web-math/open-web-math \
  --local-dir-use-symlinks False
