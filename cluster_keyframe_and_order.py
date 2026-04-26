import argparse
import json
import os
import pickle
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from run_inference_openai_compatible import load_video_frame_indices


def truncate_prompt(prompt: str) -> str:
    words = prompt.split()
    if len(words) > 40:
        words = words[:40]
    return " ".join(words)


def format_prompt(sample, ranking_mode: str):
    question = sample["question"].strip()
    if ranking_mode == "question_candidates" and sample.get("candidates"):
        options = " ".join(
            f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(sample["candidates"])
        )
        return f"{question} Options: {options}"
    return question


def video_frame_clustering(frame_features, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, init="k-means++", n_init=10).fit(frame_features)
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(frame_features - centers[:, np.newaxis, :], axis=2)
    closest_frames = np.argmin(distances, axis=1)
    return [int(closest_frames[cluster_idx]) for cluster_idx in range(num_clusters)]


def load_feature_pickle(feature_pickle: str):
    with open(feature_pickle, "rb") as f:
        return pickle.load(f)


def load_gt(gt_file: str):
    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_json(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def shard_items(items, num_shards: int, shard_rank: int):
    return [item for idx, item in enumerate(items) if idx % num_shards == shard_rank]


def rank_cluster_frames(model_clip, preprocess_clip, device, video_path: str, frame_indices, prompt: str):
    frames, _ = load_video_frame_indices(video_path, frame_indices)
    if not frames:
        raise ValueError(f"Failed to load cluster frames from {video_path}")

    image_input = torch.stack([preprocess_clip(img) for img in frames]).to(device)
    text_input = clip.tokenize([truncate_prompt(prompt)]).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(image_input)
        text_features = model_clip.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).squeeze(-1)

    sorted_indices = torch.argsort(similarity, descending=True).tolist()
    return [frame_indices[idx] for idx in sorted_indices]


def generate_keyframes(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model_clip, preprocess_clip = clip.load(args.clip_model, device=device)

    gt_data = load_gt(args.gt_file)
    shard_samples = shard_items(gt_data, args.num_shards, args.shard_rank)
    feature_data = load_feature_pickle(args.feature_pickle)
    output = load_existing_json(args.output_json) if args.resume else {}
    pending_samples = [sample for sample in shard_samples if sample["question_id"] not in output]

    for idx, sample in enumerate(tqdm(pending_samples, desc="Cluster and rank keyframes")):
        question_id = sample["question_id"]
        video_name = sample["video_name"]
        prompt = format_prompt(sample, args.ranking_mode)

        if video_name not in feature_data:
            raise KeyError(f"Missing DINO features for {video_name}")

        item = feature_data[video_name]
        frame_features = np.asarray(item["features"], dtype=np.float32)
        frame_indices = [int(i) for i in item["frame_indices"]]
        if len(frame_features) == 0 or len(frame_indices) == 0:
            raise ValueError(f"Empty feature entry for {video_name}")

        num_clusters = min(args.num_clusters, len(frame_features))
        cluster_feature_indices = video_frame_clustering(frame_features, num_clusters)
        cluster_frame_indices = [frame_indices[idx_] for idx_ in cluster_feature_indices]
        ranked_frame_indices = rank_cluster_frames(
            model_clip=model_clip,
            preprocess_clip=preprocess_clip,
            device=device,
            video_path=item["video_path"],
            frame_indices=cluster_frame_indices,
            prompt=prompt,
        )
        output[question_id] = [[int(frame_idx), order] for order, frame_idx in enumerate(ranked_frame_indices)]
        if args.flush_every > 0 and ((idx + 1) % args.flush_every == 0):
            save_json(args.output_json, output)

    save_json(args.output_json, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--feature_pickle", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_clusters", type=int, default=6)
    parser.add_argument(
        "--ranking_mode",
        choices=["question_only", "question_candidates"],
        default="question_only",
    )
    parser.add_argument("--clip_model", default="ViT-L/14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--flush_every", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate_keyframes(parse_args())
