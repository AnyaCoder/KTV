# Structure

Purpose: Utility scripts for dataset preparation, download, and benchmark file conversion.

- `download_http_ranges.py`: resumable ranged HTTP downloader for large public archives.
- `extract_zip_selected.py`: selective ZIP extractor that only expands requested members.
- `launch_qwen35_swarm.sh`: launcher for multi-replica local Qwen3.5 routing services with configurable GPU groups.
- `run_ktv_preprocess_multigpu.py`: multi-GPU wrapper that shards KTV DINO feature extraction and keyframe clustering, then merges shard outputs.
- `data/`: existing dataset conversion scripts for benchmark QA files.
