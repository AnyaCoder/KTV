# Structure

Purpose: Repository root for KTV training-free video QA code and local evaluation helpers.

- `README.md`: original setup and inference notes for the KTV codebase.
- `run_inference_multiple_choice_qa.py`: primary local LLaVA-based multiple-choice inference entrypoint.
- `run_inference_openai_compatible.py`: shared OpenAI-compatible video inference helper with STAR clip resolution.
- `run_eval_grouped_openai_compatible.py`: grouped-per-video OpenAI-compatible evaluator.
- `run_eval_star_openai_compatible.py`: grouped STAR clip evaluator using raw video segments.
- `run_eval_cached_singleq_openai.py`: single-question evaluator that reuses decoded frames per video.
- `run_eval_three_stage_openai.py`: three-stage global-to-local evaluator with router-driven local refinement.
- `dataset.py`: shared frame loading utilities for videos and extracted frame folders.
- `cluster_keyframe_and_order.py`: keyframe clustering and ordering utilities used by the original KTV workflow.
- `keyframe_select_new.py`: feature extraction and keyframe preparation script for KTV.
- `eval/`: evaluation scripts for multiple-choice outputs.
- `scripts/`: small utility scripts for dataset preparation and download/extraction.
- `ktv/llava/`: vendored LLaVA package configuration used by the original KTV implementation.
- `playground/gt_qa_files/`: benchmark question files bundled with the repository.
