# Maintenance Log

## 2026-04-26 - Promote the 0.62 segment-tree line to the default path

- Updated `run_eval_swarm_openai.py` so the default evaluator path is now the `segment_tree + keyframe_windows + soft_temporal_chain + representative_temporal_guard` configuration.
- Changed the mainline keyframe leaf path to fail loudly when keyframe metadata is missing or falls outside the clip, instead of silently reverting to uniform sampled leaves.
- Kept the older leaf, payload, and summary variants as explicit ablation flags rather than the default behavior.
- Did this to make the code match the current paper story and preserve pipeline purity for the mainline method.

## 2026-04-26 - Document the 0.62 segment-tree method line

- Added `seg_tree_method_062.md` to formalize the current strongest segment-tree pipeline in paper-style language.
- Split the method into its main contribution path, ablation-only options, and heuristic-heavy engineering pieces so later writing does not overclaim parser fallbacks or exploratory branches.
- Did this to keep the `0.62` line easy to explain and compare against later variants without confusing it with the unrelated `largest_gap` exploration state on the working branch.

## 2026-04-26 - Lock 0.62 sparse-temporal-guard checkpoint

- Recorded that the current strongest 100-sample result is the `keyframe_windows + soft_temporal_chain + representative_temporal_guard` configuration.
- The corresponding run is `data/star_dataset/out/star_rand100_seed2026_seg_tree_next_sparseguard.jsonl`, which reached `62/100 = 0.62` with an average of `6.82` final frames per question.
- Kept this as the next checkpoint before exploring further lightweight temporal-coverage refinements.

## 2026-04-26 - Add sparse temporal-guard payload on the next-step branch

- Updated `run_eval_swarm_openai.py` with a new `representative_temporal_guard` payload mode.
- This mode keeps the strong `representative_only` routing path, then sparsely adds up to a small number of boundary frames only when the evidence plan contains temporal-chain signals such as multiple stages or before/after relations.
- Did this to test whether a very light temporal guard can recover some sequencing evidence without falling back to the much heavier `node_summary` payload that raised image count but underperformed.

## 2026-04-26 - Lock 0.60 paper-fusion ablation checkpoint

- Recorded that the strongest 100-sample result so far on the paper-fusion line is the `keyframe_windows + soft_temporal_chain + representative_only` configuration.
- The corresponding run is `data/star_dataset/out/star_rand100_seed2026_seg_tree_ablate_repronly.jsonl`, which reached `60/100 = 0.60` with an average of `5.11` final frames per question.
- Kept this as the checkpoint to branch from before trying the next refinement step.

## 2026-04-26 - Add paper-fusion segment-tree options on a dedicated branch

- Updated `run_eval_swarm_openai.py` to support KTV-style keyframe-window leaves via `--key_frame_path` and `--segment_tree_leaf_source keyframe_windows`, reusing chronological cluster anchors as non-uniform temporal partitions.
- Added boundary-plus-representative node summaries and a `node_summary` payload mode so each tree node can carry compact start/representative/end evidence instead of a single frame.
- Extended the evidence-plan prompt with an optional soft temporal-chain style that records short stage labels and feeds them back into the segment-tree expansion prompt.
- Did this on the dedicated `feature/seg-tree-paper-fusion` branch so the earlier 59%-result branch remains untouched while the KTV/HiTeA/NeuS-QA fusion idea is explored separately.

## 2026-04-26 - Rebalance swarm segment-tree routing against overcompression

- Updated `run_eval_swarm_openai.py` so the evidence-plan prompt is shorter and less checklist-like, aiming for a soft grounded routing note instead of exhaustive preservation instructions.
- Tightened the segment-tree expansion prompt to expand only for concrete visible risks such as temporal-boundary ambiguity, identity ambiguity, or answer-relevant state changes.
- Added a question-aware adjacent frontier pruning pass for representative-mode segment-tree routing, plus a temporal-coverage guard that restores the first and last frontier nodes if pruning collapses everything to one frame.
- Recorded new frontier-pruning metadata in the output JSONL for later analysis.
- Did this to recover the earlier `question-aware + anti-overcompression guard` behavior inside the current swarm pipeline without introducing a separate legacy branch.

## 2026-04-26 - Harden swarm parsing for small local VLM routing

- Updated `run_eval_swarm_openai.py` so evidence-plan parsing can recover from truncated or non-strict-JSON small-model outputs, and segment-tree parsing now accepts free-form `Representative/Expand` responses.
- Updated `run_inference_openai_compatible.py` so OpenAI-compatible requests can pass through `response_format`, then enabled JSON-object constrained decoding for swarm-side small-model routing calls.
- Tightened the evidence-plan prompt so `relations` is capped to a small fixed vocabulary and added a minimal synthesized `preserve` fallback when the small model truncates after the early JSON fields.
- Raised the default evidence-plan token budget from `256` to `512` for the swarm evaluator.
- Did this to make the local `Qwen3.5-0.8B` multimodal swarm usable without changing the core segment-tree pipeline.

## 2026-04-26 - Generalize local SGLang launcher for TP groups

- Updated `scripts/launch_qwen35_swarm.sh` to accept semicolon-separated GPU groups, infer or validate `tensor-parallel-size`, and optionally enable `--sleep-on-idle`.
- Did this to support 2-GPU-per-replica experiments such as running four `Qwen3.5-2B` local small-model services across eight GPUs while reducing idle CPU overhead.

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
