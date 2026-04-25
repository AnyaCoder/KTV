import argparse
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from run_inference_openai_compatible import (
    infer_one,
    resolve_question_overview_frames,
    resolve_video_and_frames,
)


def process_video(video_dir, video_name, samples, args):
    records = []
    for sample in samples:
        try:
            if args.key_frame_data is not None:
                frames, _, _ = resolve_question_overview_frames(
                    video_dir=video_dir,
                    video_name=video_name,
                    num_frames=args.num_frames,
                    question_id=sample["question_id"],
                    keyframe_data=args.key_frame_data,
                )
            else:
                frames, _ = resolve_video_and_frames(video_dir, video_name, args.num_frames)
        except (FileNotFoundError, ValueError, KeyError) as e:
            return video_name, [], str(e)

        if not frames:
            return video_name, [], f"empty frames for video: {video_name}"

        pred = infer_one(
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
            video_frames=frames,
            question=sample["question"],
            candidates=sample["candidates"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        records.append(
            {
                "task_name": sample["task_name"],
                "question": sample["question"],
                "id": sample["question_id"],
                "answer_number": int(sample["answer_number"]),
                "candidates": sample["candidates"],
                "answer": sample["answer"],
                "pred": pred,
            }
        )
    return video_name, records, None


def main(args):
    with open(args.gt_file, "r") as f:
        data = json.load(f)

    grouped = defaultdict(list)
    for sample in data:
        grouped[sample["video_name"]].append(sample)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    error_path = os.path.join(args.output_dir, f"{args.output_name}.errors.log")

    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])

    video_items = []
    for video_name, samples in grouped.items():
        pending = [sample for sample in samples if sample["question_id"] not in done_ids]
        if pending:
            video_items.append((video_name, pending))

    lock = threading.Lock()
    progress = tqdm(total=sum(len(samples) for _, samples in video_items))

    with open(output_path, "a", encoding="utf-8") as out_f, open(
        error_path, "a", encoding="utf-8"
    ) as err_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_video, args.video_dir, video_name, samples, args): (
                    video_name,
                    len(samples),
                )
                for video_name, samples in video_items
            }
            for future in as_completed(futures):
                video_name, n_samples = futures[future]
                try:
                    _, records, error = future.result()
                    with lock:
                        if error:
                            err_f.write(error + "\n")
                        for record in records:
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        err_f.flush()
                except Exception as e:
                    with lock:
                        err_f.write(f"{video_name}: {type(e).__name__}: {e}\n")
                        err_f.flush()
                finally:
                    progress.update(n_samples)
    progress.close()


def load_keyframe_data(key_frame_path: str | None):
    if not key_frame_path:
        return None
    with open(key_frame_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--key_frame_path", default=None)
    parser.add_argument("--api_base", default="http://10.126.62.90:8003/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.key_frame_data = load_keyframe_data(args.key_frame_path)
    main(args)
