import argparse
import base64
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm

from run_inference_openai_compatible import load_video_segment, parse_star_clip_name


def image_to_data_url(image: Image.Image, format_: str = "JPEG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format_)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_grouped_prompt(samples):
    blocks = []
    for sample in samples:
        options = "\n".join(
            f"  ({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(sample["candidates"])
        )
        blocks.append(
            f"question_id: {sample['question_id']}\n"
            f"question: {sample['question']}\n"
            f"options:\n{options}"
        )

    return (
        "You are a helpful expert in video analysis.\n"
        "The input consists of a sequence of key frames from one short video clip.\n"
        "Answer every multiple-choice question for this same clip.\n"
        "Return only a JSON object whose keys are question_id and whose values are a single option letter like A, B, C, or D.\n\n"
        + "\n\n".join(blocks)
    )


def parse_response(text, samples):
    text = text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(text[start : end + 1])
            result = {}
            for sample in samples:
                value = str(parsed.get(sample["question_id"], "")).strip().upper()
                if value:
                    result[sample["question_id"]] = value[0]
            return result
    except Exception:
        pass

    result = {}
    for sample in samples:
        pattern = re.compile(rf"{re.escape(sample['question_id'])}\s*[:=]\s*([A-D])", re.I)
        match = pattern.search(text)
        if match:
            result[sample["question_id"]] = match.group(1).upper()
    if len(samples) == 1 and not result:
        match = re.search(r"\b([A-D])\b", text.upper())
        if match:
            result[samples[0]["question_id"]] = match.group(1)
    return result


def infer_clip(api_base, api_key, model_name, frames, samples, max_tokens, temperature):
    content = [{"type": "text", "text": build_grouped_prompt(samples)}]
    for image in frames:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})

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
    text = response.json()["choices"][0]["message"]["content"].strip()
    return text, parse_response(text, samples)


def process_clip(video_dir, clip_name, samples, args):
    parsed = parse_star_clip_name(clip_name)
    if parsed is None:
        raise ValueError(f"Invalid STAR clip name: {clip_name}")
    raw_name, start, end = parsed
    raw_path = os.path.join(video_dir, raw_name)
    if not os.path.exists(raw_path):
        return clip_name, [], f"missing raw video: {raw_path}"

    frames, _ = load_video_segment(raw_path, start, end, args.num_frames)
    if not frames:
        return clip_name, [], f"empty frames for clip: {clip_name}"

    raw_text, answers = infer_clip(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model_name,
        frames=frames,
        samples=samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    records = []
    for sample in samples:
        pred = answers.get(sample["question_id"], "")
        if not pred:
            pred = raw_text.strip()
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
    return clip_name, records, None


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

    clip_items = []
    for clip_name, samples in grouped.items():
        pending = [sample for sample in samples if sample["question_id"] not in done_ids]
        if pending:
            clip_items.append((clip_name, pending))

    lock = threading.Lock()
    progress = tqdm(total=len(clip_items))

    with open(output_path, "a", encoding="utf-8") as out_f, open(
        error_path, "a", encoding="utf-8"
    ) as err_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_clip, args.video_dir, clip_name, samples, args): clip_name
                for clip_name, samples in clip_items
            }
            for future in as_completed(futures):
                clip_name = futures[future]
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
                        err_f.write(f"{clip_name}: {type(e).__name__}: {e}\n")
                        err_f.flush()
                finally:
                    progress.update(1)

    progress.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", default="playground/gt_qa_files/STAR/val_qa.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--api_base", default="http://10.126.62.90:8003/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
