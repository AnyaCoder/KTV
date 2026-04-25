# Maintenance Log

## 2026-04-25 - Ignore local weight checkpoints

- Updated `.gitignore` to exclude local `*.pth` files alongside wheel artifacts.
- Did this to keep downloaded DINOv2 and other model checkpoints out of the cleaned repository checkpoints before pushing.

## 2026-04-25 - Add OpenAI-compatible eval helpers and environment checkpoint

- Added OpenAI-compatible inference and evaluation entrypoints at the repository root for remote VLM-based STAR evaluation.
- Updated `run_inference_multiple_choice_qa.py` and `ktv/llava/pyproject.toml` to support newer Torch versions plus optional flash-attn and quantized loading flags.
- Added lightweight dataset utility scripts under `scripts/` and a single-question evaluator under `eval/`.
- Added repository-level ignore rules so local datasets, caches, PDFs, and wheel files stay out of version control.

## 2026-04-25 - Start three-stage global-to-local evaluation flow

- Expanded `run_inference_openai_compatible.py` with reusable chat and temporal-window loading helpers instead of adding a separate transport layer.
- Added `run_eval_three_stage_openai.py` as a minimal three-stage runner: global overview, routing, and optional local refinement.
- Did this to start the small-large collaborative framework while reusing the existing OpenAI-compatible video QA pipeline and keeping the CLI narrow.

## 2026-04-25 - Switch local refinement from uniform windows to anchor-centered segments

- Replaced fixed uniform local windows with segments centered on the global overview anchors chosen by the router.
- Did this to align refinement with global keyframe structure instead of forcing the router to choose among arbitrary front/middle/back bins.

## 2026-04-25 - Reuse KTV clustered keyframes for global overview

- Added optional `key_frame_path` support to the three-stage runner so overview frames can come from the original KTV clustered keyframe files keyed by `question_id`.
- Reused those anchor times to build local refinement segments around actual overview keyframes instead of synthetic uniform anchors.

## 2026-04-25 - Purify the KTV keyframe pipeline

- Rewrote `keyframe_select_new.py` and `cluster_keyframe_and_order.py` into parameterized CLIs while keeping the paper pipeline intact: DINOv2 frame features, KMeans cluster centers, then CLIP ranking against the question.
- Removed the mixed fallback in the three-stage runner when `--key_frame_path` is supplied; missing or invalid keyframes now fail loudly instead of silently reverting to uniform overview frames.
