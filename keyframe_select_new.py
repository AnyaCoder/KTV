import argparse
import pickle

import av
import numpy as np
import torch
from PIL import Image
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
                batch_images.append(frame.to_image())
                if len(batch_images) >= batch_size:
                    inputs = processor(images=batch_images, return_tensors="pt")
                    outputs = model(pixel_values=inputs["pixel_values"].to(device))
                    features.extend(outputs.pooler_output.cpu().numpy())
                    batch_images.clear()
            current_idx += 1
            if current_idx > max_idx:
                break

        if batch_images:
            inputs = processor(images=batch_images, return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"].to(device))
            features.extend(outputs.pooler_output.cpu().numpy())

    container.close()
    return np.asarray(features, dtype=np.float32)


def load_gt(gt_file: str):
    import json

    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_features(args):
    model = Dinov2Model.from_pretrained(args.model_name)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
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

    output = {}
    for video_name in tqdm(unique_videos, desc="Extract DINO features"):
        video_path, start, end = resolve_video_time_range(args.video_dir, video_name)
        frame_indices, fps = get_effective_frame_indices(video_path, start, end, args.max_frames)
        features = dino_feature_stream(
            video_path=video_path,
            frame_indices=frame_indices,
            model=model,
            processor=processor,
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

    with open(args.output_pickle, "wb") as f:
        pickle.dump(output, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--output_pickle", required=True)
    parser.add_argument("--model_name", default="facebook/dinov2-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=5400)
    parser.add_argument("--batch_size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    extract_features(parse_args())
