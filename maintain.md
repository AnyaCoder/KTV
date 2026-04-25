# Maintenance Log

## 2026-04-25 - Add evidence-plan segment-tree routing

- Extended `run_eval_swarm_openai.py` with a two-stage segment-tree router: an optional text-only evidence-plan pass from the question and options, then evidence-plan-guided tree expansion without directly predicting the answer.
- Added payload controls to compare sending node representative frames versus all sampled frames inside the selected frontier, plus an optional final image resize for long multi-image requests.
- Did this to move from generic compression toward question-conditioned evidence preservation while keeping matched ablations available.

## 2026-04-25 - Add segment-tree VLM frame routing

- Extended `run_eval_swarm_openai.py` with a `segment_tree` selection mode that builds a binary tree over sampled frames, asks the VLM to choose each interval's representative frame, and records compact tree metadata per sample.
- Enforced monotonic frontier selection so if any subtree requires expansion, ancestors cannot collapse it back into a single representative frame.
- Did this to match the new hierarchical "choose nodes on the tree that still cover the whole video" direction without adding a separate runner.

## 2026-04-25 - Add VLM swarm frame selector

- Added `run_eval_swarm_openai.py` for sliding-window frame selection with a VLM selector and VLM-based adjacent-frame merging before final QA.
- Reused the existing OpenAI-compatible transport and STAR clip resolution helpers so the new evaluator can run against the same remote Qwen-VL endpoint.
- Did this to test an online agent-swarm-style alternative to fixed uniform sampling and KTV clustered keyframes.

## 2026-04-25 - Add repository-local coding rules skill

- Added `.codex/skills/codex-guidelines/` to turn the repository root `CODEX.md` into a reusable skill.
- Did this to preserve the repo's local coding rules in a form that can be reused directly during future coding work.

## 2026-04-25 - Stop writing doc logs inside `.codex/skills`

- Removed `maintain.md` and `structure.md` files from `.codex/skills/` and `.codex/skills/codex-guidelines/`.
- Updated the local `codex-guidelines` skill and the shared `codebase-doc-maintainer` skill to skip local skill folders by default.
- Did this because skill directories are already small and metadata-heavy, so extra folder docs add noise instead of navigation value.

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
