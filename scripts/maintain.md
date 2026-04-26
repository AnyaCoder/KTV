# Maintenance Log

## 2026-04-26 - Track multi-GPU KTV preprocessing wrapper

- Added `run_ktv_preprocess_multigpu.py` to the documented script inventory.
- Kept this wrapper alongside the root KTV preprocessing CLIs because full STAR keyframe generation now depends on sharded DINO extraction and sharded cluster-center generation.
- Did this to make the full `0.62` pipeline reproducible instead of relying on an undocumented local helper.

## 2026-04-25 - Add lightweight download and extraction helpers

- Added `download_http_ranges.py` for parallel ranged downloads from reachable HTTP endpoints.
- Added `extract_zip_selected.py` to extract only benchmark-required videos from downloaded archives.
- Did this to keep dataset setup practical on hosts with unstable access to some upstream sources.
