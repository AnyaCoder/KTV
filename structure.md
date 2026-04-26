# Structure

Purpose: Repository root for KTV training-free video QA code and local evaluation helpers.

- `README.md`: original setup and inference notes for the KTV codebase.
- `.gitignore`: repository-local ignore rules for datasets, logs, wheels, and local checkpoint weights.
- `run_inference_multiple_choice_qa.py`: primary local LLaVA-based multiple-choice inference entrypoint.
- `run_inference_openai_compatible.py`: shared OpenAI-compatible video inference helper with STAR clip resolution.
- `run_eval_grouped_openai_compatible.py`: grouped-per-video OpenAI-compatible evaluator.
- `run_eval_star_openai_compatible.py`: grouped STAR clip evaluator using raw video segments.
- `run_eval_cached_singleq_openai.py`: single-question evaluator that reuses decoded frames per video.
- `run_eval_three_stage_openai.py`: three-stage global-to-local evaluator with router-selected anchor frames, optional KTV keyframe reuse, and local refinement.
- `run_eval_swarm_openai.py`: small-large swarm evaluator whose default path is the `0.62` KTV-keyframe-window + soft temporal chain + sparse temporal guard segment-tree pipeline, while older routing and payload variants remain as ablation flags.
- `seg_tree_method_062.md`: paper-style note that defines the current best `0.62` segment-tree pipeline, its method core, and its ablation-only components.
- `dataset.py`: shared frame loading utilities for videos and extracted frame folders.
- `cluster_keyframe_and_order.py`: keyframe clustering and ordering utilities used by the original KTV workflow.
- `keyframe_select_new.py`: CLI for extracting DINOv2 frame features over each evaluation sample's effective video range.
- `cluster_keyframe_and_order.py`: CLI for KMeans keyframe selection plus CLIP-based question ordering that outputs `question_id -> keyframes`.
- `eval/`: evaluation scripts for multiple-choice outputs.
- `scripts/`: utility scripts for dataset preparation plus local multi-replica SGLang launchers.
- `ktv/llava/`: vendored LLaVA package configuration used by the original KTV implementation.
- `playground/gt_qa_files/`: benchmark question files bundled with the repository.
