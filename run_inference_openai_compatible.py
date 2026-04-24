import argparse
import base64
import json
import os
from io import BytesIO

import cv2
import requests
from PIL import Image
from tqdm import tqdm

from dataset import load_video
from utils import get_chunk


def image_to_data_url(image: Image.Image, format_: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format_)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format_.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def build_prompt(question: str, candidates) -> str:
    options = "\n".join(
        f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates)
    )
    return (
        "You are a helpful expert in video analysis.\n"
        "The input consists of a sequence of key frames from a video.\n"
        "Answer the multiple-choice question by returning only the best option letter.\n"
        f"Question: {question}\n"
        f"Options:\n{options}\n"
        "Answer:"
    )


def build_content(prompt_text: str, video_frames):
    content = [{"type": "text", "text": prompt_text}]
    for image in video_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image)},
            }
        )
    return content


def parse_star_clip_name(video_name: str):
    if not video_name.endswith(".mp4"):
        return None
    stem = video_name[:-4]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        start = float(parts[-2])
        end = float(parts[-1])
    except ValueError:
        return None
    raw_name = "_".join(parts[:-2]) + ".mp4"
    return raw_name, start, end


def load_video_segment(video_path: str, start: float, end: float, num_frms: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS for video {video_path}")

    clip_start = max(0, min(int(start * fps), total_num_frames - 1))
    clip_end = max(clip_start + 1, min(int(end * fps), total_num_frames))
    interval = (clip_end - clip_start) / max(num_frms, 1)
    frame_idx = [int(clip_start + i * interval) for i in range(num_frms)]

    clip_imgs = []
    original_sizes = []
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        clip_imgs.append(img)
        original_sizes.append(img.size)
    cap.release()
    return clip_imgs, tuple(original_sizes)


def load_video_frame_indices(video_path: str, frame_indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    clip_imgs = []
    original_sizes = []
    for idx in sorted(set(int(i) for i in frame_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        clip_imgs.append(img)
        original_sizes.append(img.size)
    cap.release()
    return clip_imgs, tuple(original_sizes)


def get_video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS for video {video_path}")
    return total_num_frames / fps


def resolve_video_time_range(video_dir: str, video_name: str):
    video_path, start, end = resolve_video_source(video_dir, video_name)
    if start is None or end is None:
        start = 0.0
        end = get_video_duration_seconds(video_path)
    return video_path, start, end


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS for video {video_path}")
    return fps


def build_anchor_windows(start: float, end: float, anchor_times):
    if not anchor_times:
        raise ValueError("anchor_times must not be empty")
    num_anchors = len(anchor_times)
    anchor_times = [max(start, min(end, float(t))) for t in anchor_times]
    anchor_times.sort()
    if num_anchors == 1:
        return [(start, end)]

    boundaries = [start]
    for idx in range(num_anchors - 1):
        boundaries.append((anchor_times[idx] + anchor_times[idx + 1]) / 2.0)
    boundaries.append(end)
    windows = []
    for idx in range(num_anchors):
        left = max(start, boundaries[idx])
        right = min(end, boundaries[idx + 1])
        if right <= left:
            right = min(end, left + 1e-6)
        windows.append((left, right))
    return windows


def load_question_keyframes(keyframe_data, question_id: str):
    if not keyframe_data:
        return None
    keyframes = keyframe_data.get(question_id)
    if not keyframes:
        return None
    indices = []
    for item in keyframes:
        if isinstance(item, (list, tuple)) and item:
            try:
                indices.append(int(item[0]))
            except (TypeError, ValueError):
                continue
        else:
            try:
                indices.append(int(item))
            except (TypeError, ValueError):
                continue
    if not indices:
        return None
    return sorted(set(indices))


def resolve_question_overview_frames(
    video_dir: str,
    video_name: str,
    num_frames: int,
    question_id: str | None = None,
    keyframe_data=None,
):
    video_path, start, end = resolve_video_time_range(video_dir, video_name)
    fps = get_video_fps(video_path)
    clip_start_frame = int(start * fps)
    clip_end_frame = max(clip_start_frame + 1, int(end * fps))

    keyframe_indices = load_question_keyframes(keyframe_data, question_id) if question_id else None
    if keyframe_indices:
        filtered = [idx for idx in keyframe_indices if clip_start_frame <= idx < clip_end_frame]
        if filtered:
            frames, sizes = load_video_frame_indices(video_path, filtered[:num_frames])
            anchor_times = [idx / fps for idx in filtered[: len(frames)]]
            if frames:
                return frames, sizes, anchor_times

    frames, sizes = resolve_video_and_frames(video_dir, video_name, num_frames)
    anchor_times = []
    if frames:
        span = max(end - start, 1e-6)
        interval = span / max(len(frames), 1)
        anchor_times = [start + interval * idx for idx in range(len(frames))]
    return frames, sizes, anchor_times


def resolve_video_anchor_and_frames_from_times(
    video_dir: str, video_name: str, num_frames: int, anchor_idx: int, anchor_times
):
    if not anchor_times:
        raise ValueError("anchor_times must not be empty")
    if anchor_idx < 0 or anchor_idx >= len(anchor_times):
        raise ValueError(f"anchor_idx must be in [0, {len(anchor_times)}), got {anchor_idx}")

    video_path, start, end = resolve_video_time_range(video_dir, video_name)
    anchor_windows = build_anchor_windows(start, end, anchor_times)
    window_start, window_end = anchor_windows[anchor_idx]
    return load_video_segment(video_path, window_start, window_end, num_frames)


def build_uniform_anchor_times(start: float, end: float, num_anchors: int):
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if num_anchors == 1:
        return [start]
    span = max(end - start, 1e-6)
    return [start + span * idx / num_anchors for idx in range(num_anchors)]


def resolve_video_source(video_dir: str, video_name: str):
    direct_path = os.path.join(video_dir, video_name)
    if os.path.exists(direct_path):
        return direct_path, None, None

    star_clip = parse_star_clip_name(video_name)
    if star_clip is not None:
        raw_name, start, end = star_clip
        raw_path = os.path.join(video_dir, raw_name)
        if os.path.exists(raw_path):
            return raw_path, start, end

    raise FileNotFoundError(f"Cannot resolve video for {video_name} under {video_dir}")


def resolve_video_and_frames(video_dir: str, video_name: str, num_frames: int):
    video_path, start, end = resolve_video_source(video_dir, video_name)
    if start is None or end is None:
        return load_video(video_path, keyframe=None, num_frms=num_frames)
    return load_video_segment(video_path, start, end, num_frames)


def resolve_video_anchor_and_frames(
    video_dir: str, video_name: str, num_frames: int, anchor_idx: int, num_anchors: int
):
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    if anchor_idx < 0 or anchor_idx >= num_anchors:
        raise ValueError(f"anchor_idx must be in [0, {num_anchors}), got {anchor_idx}")

    video_path, start, end = resolve_video_time_range(video_dir, video_name)
    anchor_windows = build_anchor_windows(start, end, build_uniform_anchor_times(start, end, num_anchors))
    window_start, window_end = anchor_windows[anchor_idx]
    return load_video_segment(video_path, window_start, window_end, num_frames)


def chat_with_frames(
    api_base: str,
    api_key: str,
    model_name: str,
    video_frames,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
):
    content = build_content(prompt_text, video_frames)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def infer_one(
    api_base: str,
    api_key: str,
    model_name: str,
    video_frames,
    question: str,
    candidates,
    max_tokens: int,
    temperature: float,
):
    return chat_with_frames(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        video_frames=video_frames,
        prompt_text=build_prompt(question, candidates),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_inference(args):
    with open(args.gt_file, "r") as f:
        gt_qa_pairs = json.load(f)
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    generated_id = set()
    mode = "a" if os.path.exists(output_path) else "w"
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                generated_id.add(json.loads(line)["id"])

    with open(output_path, mode) as ans_file:
        for sample in tqdm(gt_qa_pairs):
            question_id = sample["question_id"]
            if question_id in generated_id:
                continue

            try:
                video_frames, _ = resolve_video_and_frames(
                    args.video_dir, sample["video_name"], args.num_frames
                )
            except FileNotFoundError as e:
                print(str(e))
                continue
            output = infer_one(
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                video_frames=video_frames,
                question=sample["question"],
                candidates=sample["candidates"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            record = {
                "task_name": sample["task_name"],
                "question": sample["question"],
                "id": question_id,
                "answer_number": sample["answer_number"],
                "candidates": sample["candidates"],
                "answer": sample["answer"],
                "pred": output,
            }
            print(output)
            ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--api_base", default="http://10.126.62.90:8003/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
