# Maintenance Log

## 2026-04-25 - Add single-question prediction evaluator

- Added `eval_multiple_choice_single.py` to score line-delimited prediction files where each question is answered independently.
- Did this to support the OpenAI-compatible single-question evaluation flow without changing the original grouped evaluator.
