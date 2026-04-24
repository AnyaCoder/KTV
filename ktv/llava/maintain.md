# Maintenance Log

## 2026-04-25 - Relax Torch version pins for local environment setup

- Updated `pyproject.toml` to accept Torch 2.4.x and matching torchvision versions.
- Did this to keep the local environment compatible with the installed CUDA 12 / flash-attn setup without patching downstream code paths.
