# Maintenance Log

## 2026-04-25 - Add lightweight download and extraction helpers

- Added `download_http_ranges.py` for parallel ranged downloads from reachable HTTP endpoints.
- Added `extract_zip_selected.py` to extract only benchmark-required videos from downloaded archives.
- Did this to keep dataset setup practical on hosts with unstable access to some upstream sources.
