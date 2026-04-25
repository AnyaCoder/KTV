# Maintenance Log

## 2026-04-25 - Ignore local weight checkpoints

- Updated `.gitignore` to exclude local `*.pth` files alongside wheel artifacts.
- Did this to keep downloaded DINOv2 and other model checkpoints out of the cleaned repository checkpoints before pushing.

## 2026-04-25 - Add OpenAI-compatible eval helpers and environment checkpoint

- Added OpenAI-compatible inference and evaluation entrypoints at the repository root for remote VLM-based STAR evaluation.
- Updated `run_inference_multiple_choice_qa.py` and `ktv/llava/pyproject.toml` to support newer Torch versions plus optional flash-attn and quantized loading flags.
- Added lightweight dataset utility scripts under `scripts/` and a single-question evaluator under `eval/`.
- Added repository-level ignore rules so local datasets, caches, PDFs, and wheel files stay out of version control.
