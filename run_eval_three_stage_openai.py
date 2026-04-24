import argparse
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from run_inference_openai_compatible import (
    chat_with_frames,
    resolve_question_overview_frames,
    resolve_video_anchor_and_frames,
    resolve_video_anchor_and_frames_from_times,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" `\"'()[]{}.,:;")


def normalize_option_answer(text: str, candidates) -> str:
    if not isinstance(text, str):
        return ""
    allowed = [chr(ord("A") + idx) for idx in range(len(candidates))]
    stripped = text.strip()
    upper = stripped.upper()

    if upper[:1] in allowed and (len(upper) == 1 or upper[1] in ").: -"):
        return upper[0]

    norm_text = normalize_text(stripped)
    candidate_pairs = [
        (chr(ord("A") + idx), normalize_text(candidate)) for idx, candidate in enumerate(candidates)
    ]
    for letter, norm_candidate in candidate_pairs:
        if norm_text == norm_candidate:
            return letter

    for letter, norm_candidate in sorted(candidate_pairs, key=lambda item: len(item[1]), reverse=True):
        if norm_candidate and norm_candidate in norm_text:
            return letter

    match = re.search(r"\b([A-F])\b", upper)
    if match and match.group(1) in allowed:
        return match.group(1)
    return ""


def parse_json_object(text: str):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_router_prompt(question: str, candidates, num_anchors: int) -> str:
    options = "\n".join(
        f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates)
    )
    return (
        "You are the routing stage of a three-stage video QA system.\n"
        "The frames are a global overview sampled across the whole video.\n"
        "Each overview frame acts as an anchor for one local refinement segment around that moment.\n"
        "Give a tentative answer, estimate confidence, and choose one anchor frame index "
        "for local refinement if more detail is needed.\n"
        "Return JSON only with keys answer, confidence, and focus_window.\n"
        "answer must be a single option letter such as A, B, C, or D.\n"
        f"confidence must be one of: high, medium, low.\n"
        f"focus_window must be an integer between 0 and {num_anchors - 1}.\n"
        f"Question: {question}\n"
        f"Options:\n{options}\n"
    )


def build_solver_prompt(question: str, candidates, has_local_frames: bool) -> str:
    options = "\n".join(
        f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates)
    )
    if has_local_frames:
        frame_note = (
            "The input consists of overview frames from the whole video followed by detailed "
            "frames from one relevant short segment. Use both the global context and the local "
            "evidence."
        )
    else:
        frame_note = "The input consists of overview frames sampled from the whole video."
    return (
        "You are a helpful expert in video analysis.\n"
        f"{frame_note}\n"
        "Answer the multiple-choice question by returning only the best option letter.\n"
        f"Question: {question}\n"
        f"Options:\n{options}\n"
        "Answer:"
    )


def parse_router_output(raw_text: str, candidates, num_anchors: int):
    parsed = parse_json_object(raw_text) or {}

    answer = normalize_option_answer(str(parsed.get("answer", "")), candidates)
    if not answer:
        answer = normalize_option_answer(raw_text, candidates)

    confidence = str(parsed.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        match = re.search(r"\b(high|medium|low)\b", raw_text.lower())
        confidence = match.group(1) if match else "low"

    try:
        focus_window = int(parsed.get("focus_window", num_anchors // 2))
    except (TypeError, ValueError):
        focus_window = num_anchors // 2
    focus_window = max(0, min(num_anchors - 1, focus_window))

    return {
        "answer": answer,
        "confidence": confidence,
        "focus_window": focus_window,
        "raw_text": raw_text.strip(),
    }


def should_refine(router_result, always_refine: bool) -> bool:
    if always_refine:
        return True
    return router_result["confidence"] != "high" or not router_result["answer"]


def resolve_router_params(args):
    return {
        "api_base": args.router_api_base or args.api_base,
        "api_key": args.router_api_key or args.api_key,
        "model_name": args.router_model_name or args.model_name,
        "max_tokens": args.router_max_tokens,
        "temperature": args.router_temperature,
    }


def load_keyframe_data(key_frame_path: str | None):
    if not key_frame_path:
        return None
    with open(key_frame_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_video(video_dir, video_name, samples, args):
    router_params = resolve_router_params(args)
    local_cache = {}
    overview_cache = {}
    records = []

    for sample in samples:
        overview_key = sample["question_id"] if args.key_frame_data else "__shared__"
        if overview_key not in overview_cache:
            try:
                overview_cache[overview_key] = resolve_question_overview_frames(
                    video_dir=video_dir,
                    video_name=video_name,
                    num_frames=args.global_num_frames,
                    question_id=sample["question_id"],
                    keyframe_data=args.key_frame_data,
                )
            except (FileNotFoundError, ValueError, KeyError) as e:
                return video_name, [], str(e)

        global_frames, _, anchor_times = overview_cache[overview_key]
        if not global_frames:
            return video_name, [], f"empty global frames for video: {video_name}"
        num_anchors = len(global_frames)

        router_raw = chat_with_frames(
            api_base=router_params["api_base"],
            api_key=router_params["api_key"],
            model_name=router_params["model_name"],
            video_frames=global_frames,
            prompt_text=build_router_prompt(
                sample["question"], sample["candidates"], num_anchors
            ),
            max_tokens=router_params["max_tokens"],
            temperature=router_params["temperature"],
        )
        router_result = parse_router_output(
            router_raw, candidates=sample["candidates"], num_anchors=num_anchors
        )

        refined = should_refine(router_result, args.always_refine)
        pred = router_result["answer"]

        if refined:
            focus_window = router_result["focus_window"]
            cache_key = (overview_key, focus_window)
            if cache_key not in local_cache:
                try:
                    if args.key_frame_data and anchor_times:
                        local_frames, _ = resolve_video_anchor_and_frames_from_times(
                            video_dir,
                            video_name,
                            args.local_num_frames,
                            focus_window,
                            anchor_times,
                        )
                    else:
                        local_frames, _ = resolve_video_anchor_and_frames(
                            video_dir, video_name, args.local_num_frames, focus_window, num_anchors
                        )
                except (FileNotFoundError, ValueError, KeyError) as e:
                    return video_name, [], str(e)
                local_cache[cache_key] = local_frames

            local_frames = local_cache[cache_key]
            merged_frames = list(global_frames) + list(local_frames)
            pred = chat_with_frames(
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                video_frames=merged_frames,
                prompt_text=build_solver_prompt(
                    sample["question"], sample["candidates"], has_local_frames=True
                ),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ).strip()
        elif pred:
            pred = pred.strip()
        else:
            pred = chat_with_frames(
                api_base=args.api_base,
                api_key=args.api_key,
                model_name=args.model_name,
                video_frames=global_frames,
                prompt_text=build_solver_prompt(
                    sample["question"], sample["candidates"], has_local_frames=False
                ),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ).strip()

        records.append(
            {
                "task_name": sample["task_name"],
                "question": sample["question"],
                "id": sample["question_id"],
                "answer_number": int(sample["answer_number"]),
                "candidates": sample["candidates"],
                "answer": sample["answer"],
                "pred": pred,
                "router_pred": router_result["answer"],
                "router_confidence": router_result["confidence"],
                "router_focus_window": router_result["focus_window"],
                "router_raw": router_result["raw_text"],
                "refined": refined,
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
    output_path = f"{args.output_dir}/{args.output_name}.jsonl"
    error_path = f"{args.output_dir}/{args.output_name}.errors.log"

    done_ids = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
    except FileNotFoundError:
        pass

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--api_base", default="http://10.126.62.90:8003/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--router_api_base", default=None)
    parser.add_argument("--router_api_key", default=None)
    parser.add_argument("--router_model_name", default=None)
    parser.add_argument("--key_frame_path", default=None)
    parser.add_argument("--global_num_frames", type=int, default=6)
    parser.add_argument("--local_num_frames", type=int, default=6)
    parser.add_argument("--router_max_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--router_temperature", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--always_refine", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.key_frame_data = load_keyframe_data(args.key_frame_path)
    main(args)
