import argparse
import os
import pickle
import sys

import av
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

from run_inference_openai_compatible import resolve_video_time_range


DEFAULT_DINOV2_REPO = ".codex_tmp/dinov2"
DEFAULT_DINOV2_WEIGHTS = "dinov2_vitl14_reg4_pretrain.pth"

fast_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def dino_feature_stream(video_path: str, frame_indices, model, processor, device, batch_size: int):
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
                batch_images.append(fast_transform(frame.to_image()))
                if len(batch_images) >= batch_size:
                    batch_tensors = torch.stack(batch_images).to(device)
                    outputs = model(batch_tensors)
                    features.extend(outputs.cpu().numpy())
                    batch_images.clear()
            current_idx += 1
            if current_idx > max_idx:
                break

        if batch_images:
            batch_tensors = torch.stack(batch_images).to(device)
            outputs = model(batch_tensors)
            features.extend(outputs.cpu().numpy())

    container.close()
    return np.asarray(features, dtype=np.float32)


def load_gt(gt_file: str):
    import json

    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_pickle(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def atomic_dump_pickle(path: str, obj):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def extract_features(args):
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError(f"shard_rank must be in [0, {args.num_shards}), got {args.shard_rank}")

    sys.path.insert(0, args.dinov2_repo)
    from dinov2.hub.backbones import dinov2_vitl14_reg

    model = dinov2_vitl14_reg(pretrained=False)
    state_dict = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    gt_data = load_gt(args.gt_file)
    unique_videos = []
    seen = set()
    for sample in gt_data:
        video_name = sample["video_name"]
        if video_name not in seen:
            seen.add(video_name)
            unique_videos.append(video_name)

    shard_videos = [
        video_name
        for idx, video_name in enumerate(unique_videos)
        if idx % args.num_shards == args.shard_rank
    ]
    output = load_existing_pickle(args.output_pickle) if args.resume else {}
    pending_videos = [video_name for video_name in shard_videos if video_name not in output]
    desc = f"Extract DINO features [shard {args.shard_rank}/{args.num_shards}]"
    pending_since_flush = 0
    for video_name in tqdm(pending_videos, desc=desc):
        video_path, start, end = resolve_video_time_range(args.video_dir, video_name)
        frame_indices, fps = get_effective_frame_indices(video_path, start, end, args.max_frames)
        features = dino_feature_stream(
            video_path=video_path,
            frame_indices=frame_indices,
            model=model,
            processor=None,
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
        pending_since_flush += 1
        if args.flush_every > 0 and pending_since_flush >= args.flush_every:
            atomic_dump_pickle(args.output_pickle, output)
            pending_since_flush = 0

    atomic_dump_pickle(args.output_pickle, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_pickle", required=True)
    parser.add_argument("--dinov2_repo", default=DEFAULT_DINOV2_REPO)
    parser.add_argument("--weights_path", default=DEFAULT_DINOV2_WEIGHTS)
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
