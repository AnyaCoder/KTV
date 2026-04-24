# Maintenance Log

## 2026-04-25 - Add OpenAI-compatible eval helpers and environment checkpoint

- Added OpenAI-compatible inference and evaluation entrypoints at the repository root for remote VLM-based STAR evaluation.
- Updated `run_inference_multiple_choice_qa.py` and `ktv/llava/pyproject.toml` to support newer Torch versions plus optional flash-attn and quantized loading flags.
- Added lightweight dataset utility scripts under `scripts/` and a single-question evaluator under `eval/`.
- Added repository-level ignore rules so local datasets, caches, PDFs, and wheel files stay out of version control.

## 2026-04-25 - Start three-stage global-to-local evaluation flow

- Expanded `run_inference_openai_compatible.py` with reusable chat and temporal-window loading helpers instead of adding a separate transport layer.
- Added `run_eval_three_stage_openai.py` as a minimal three-stage runner: global overview, routing, and optional local refinement.
- Did this to start the small-large collaborative framework while reusing the existing OpenAI-compatible video QA pipeline and keeping the CLI narrow.
