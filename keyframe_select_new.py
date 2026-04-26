import argparse
import json
import os
import pickle
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

from run_inference_openai_compatible import resolve_video_time_range


def get_effective_frame_indices(video_path: str, start: float, end: float, max_frames: int):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    if total_frames <= 0:
        container.close()
        raise ValueError(f"Cannot determine frame count for {video_path}")

    if stream.average_rate is None:
        container.close()
        raise ValueError(f"Cannot determine FPS for {video_path}")
    fps = float(stream.average_rate)
    container.close()

    clip_start = max(0, min(int(start * fps), total_frames - 1))
    clip_end = max(clip_start + 1, min(int(end * fps), total_frames))
    clip_length = clip_end - clip_start
    sample_count = min(max_frames, clip_length)
    if sample_count <= 0:
        raise ValueError(f"Empty clip range for {video_path}: {start}-{end}")

    if clip_length <= sample_count:
        indices = np.arange(clip_start, clip_end, dtype=int)
    else:
        indices = np.linspace(clip_start, clip_end - 1, sample_count, dtype=int)
    return indices.tolist(), fps


def build_local_dino_transform():
    return transforms.Compose(
        [
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_dino_backbone(args, device):
    weights_path = Path(args.weights_path).expanduser() if args.weights_path else None
    repo_path = Path(args.dinov2_repo).expanduser() if args.dinov2_repo else None
    if weights_path and weights_path.exists() and repo_path and repo_path.exists():
        model = torch.hub.load(
            str(repo_path.resolve()),
            "dinov2_vitl14_reg",
            source="local",
            weights=str(weights_path.resolve()),
        )
        model.to(device)
        model.eval()
        return {
            "kind": "local",
            "model": model,
            "transform": build_local_dino_transform(),
        }

    model = Dinov2Model.from_pretrained(args.model_name)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    return {
        "kind": "hf",
        "model": model,
        "processor": processor,
    }


def encode_batch(images, loader, device):
    if loader["kind"] == "local":
        pixel_values = torch.stack([loader["transform"](img) for img in images]).to(device)
        outputs = loader["model"].forward_features(pixel_values)
        return outputs["x_norm_clstoken"].detach().cpu().numpy()

    inputs = loader["processor"](images=images, return_tensors="pt")
    outputs = loader["model"](pixel_values=inputs["pixel_values"].to(device))
    return outputs.pooler_output.detach().cpu().numpy()


def dino_feature_stream(video_path: str, frame_indices, loader, device, batch_size: int):
    container = av.open(video_path)
    stream = container.streams.video[0]
    needed = set(frame_indices)
    max_idx = max(frame_indices)
    current_idx = 0
    batch_images = []
    features = []

    with torch.no_grad():
        for frame in container.decode(stream):
            if current_idx in needed:
                batch_images.append(frame.to_image())
                if len(batch_images) >= batch_size:
                    features.extend(encode_batch(batch_images, loader, device))
                    batch_images.clear()
            current_idx += 1
            if current_idx > max_idx:
                break

        if batch_images:
            features.extend(encode_batch(batch_images, loader, device))

    container.close()
    return np.asarray(features, dtype=np.float32)


def load_gt(gt_file: str):
    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def select_unique_videos(gt_data):
    unique_videos = []
    seen = set()
    for sample in gt_data:
        video_name = sample["video_name"]
        if video_name in seen:
            continue
        seen.add(video_name)
        unique_videos.append(video_name)
    return unique_videos


def shard_items(items, num_shards: int, shard_rank: int):
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_rank]


def load_existing_pickle(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, path)


def extract_features(args):
    device = torch.device(args.device)
    loader = load_dino_backbone(args, device)

    gt_data = load_gt(args.gt_file)
    unique_videos = select_unique_videos(gt_data)
    shard_videos = shard_items(unique_videos, args.num_shards, args.shard_rank)

    output = load_existing_pickle(args.output_pickle) if args.resume else {}
    pending_videos = [video_name for video_name in shard_videos if video_name not in output]

    for idx, video_name in enumerate(tqdm(pending_videos, desc="Extract DINO features")):
        video_path, start, end = resolve_video_time_range(args.video_dir, video_name)
        frame_indices, fps = get_effective_frame_indices(video_path, start, end, args.max_frames)
        features = dino_feature_stream(
            video_path=video_path,
            frame_indices=frame_indices,
            loader=loader,
            device=device,
            batch_size=args.batch_size,
        )
        output[video_name] = {
            "video_path": video_path,
            "fps": fps,
            "start": start,
            "end": end,
            "frame_indices": frame_indices,
            "features": features,
        }
        if args.flush_every > 0 and ((idx + 1) % args.flush_every == 0):
            save_pickle(args.output_pickle, output)

    save_pickle(args.output_pickle, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_pickle", required=True)
    parser.add_argument("--model_name", default="facebook/dinov2-large")
    parser.add_argument("--dinov2_repo", default=".codex_tmp/dinov2")
    parser.add_argument("--weights_path", default="dinov2_vitl14_reg4_pretrain.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=5400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    extract_features(parse_args())
