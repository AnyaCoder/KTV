import argparse
import json
import pickle

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

from run_inference_openai_compatible import load_video_frame_indices


def truncate_prompt(prompt: str) -> str:
    words = prompt.split()
    if len(words) > 40:
        words = words[10:50]
    return " ".join(words)


def video_frame_clustering(frame_features, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, init="k-means++", n_init=10).fit(
        frame_features
    )
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(frame_features - centers[:, np.newaxis, :], axis=2)
    closest_frames = np.argmin(distances, axis=1)
    cluster_center_indices = []
    for cluster_idx in range(num_clusters):
        _ = labels == cluster_idx
        cluster_center_indices.append(int(closest_frames[cluster_idx]))
    return cluster_center_indices


def load_feature_pickle(feature_pickle: str):
    with open(feature_pickle, "rb") as f:
        return pickle.load(f)


def load_gt(gt_file: str):
    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def rank_cluster_frames(model_clip, preprocess_clip, device, video_path: str, frame_indices, prompt: str):
    frames, _ = load_video_frame_indices(video_path, frame_indices)
    if not frames:
        raise ValueError(f"Failed to load cluster frames from {video_path}")

    image_input = torch.stack([preprocess_clip(img) for img in frames]).to(device)
    prompt = truncate_prompt(prompt)
    text_input = clip.tokenize([prompt]).to(device)

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
    feature_data = load_feature_pickle(args.feature_pickle)
    output = {}

    for sample in tqdm(gt_data, desc="Cluster and rank keyframes"):
        question_id = sample["question_id"]
        video_name = sample["video_name"]
        prompt = sample["question"]

        if video_name not in feature_data:
            raise KeyError(f"Missing DINO features for {video_name}")

        item = feature_data[video_name]
        frame_features = np.asarray(item["features"], dtype=np.float32)
        frame_indices = [int(i) for i in item["frame_indices"]]
        if len(frame_features) == 0 or len(frame_indices) == 0:
            raise ValueError(f"Empty feature entry for {video_name}")

        num_clusters = min(args.num_clusters, len(frame_features))
        cluster_feature_indices = video_frame_clustering(frame_features, num_clusters)
        cluster_frame_indices = [frame_indices[idx] for idx in cluster_feature_indices]
        ranked_frame_indices = rank_cluster_frames(
            model_clip=model_clip,
            preprocess_clip=preprocess_clip,
            device=device,
            video_path=item["video_path"],
            frame_indices=cluster_frame_indices,
            prompt=prompt,
        )
        output[question_id] = [[int(frame_idx), order] for order, frame_idx in enumerate(ranked_frame_indices)]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--feature_pickle", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_clusters", type=int, default=12)
    parser.add_argument("--clip_model", default="ViT-L/14")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    generate_keyframes(parse_args())
