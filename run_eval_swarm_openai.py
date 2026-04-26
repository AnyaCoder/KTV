import argparse
import json
import os
import queue
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
from PIL import Image
from tqdm import tqdm

from run_inference_openai_compatible import (
    build_anchor_windows,
    chat_with_frames,
    get_video_fps,
    infer_one,
    load_video_frame_indices,
    load_question_keyframes,
    resolve_video_source,
    resolve_video_time_range,
)

MAX_FRONTIER_LEAVES = 4
MAX_EVIDENCE_RELATIONS = 3
MAX_EVIDENCE_STAGES = 3
ALLOWED_EVIDENCE_RELATIONS = {"before", "after", "while", "same object", "state change"}


@dataclass(frozen=True)
class FrameCandidate:
    frame_idx: int
    time_sec: float
    image: Image.Image


@dataclass(frozen=True)
class SegmentTreeNode:
    node_id: str
    level: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    representative: FrameCandidate
    summary_frames: tuple[FrameCandidate, ...]
    children: tuple["SegmentTreeNode", ...]
    expand: bool

    @property
    def is_leaf(self) -> bool:
        return not self.children


class EndpointPool:
    def __init__(self, api_bases, api_keys, max_parallel_per_endpoint: int = 1):
        if not api_bases:
            raise ValueError("api_bases must not be empty")
        if len(api_bases) != len(api_keys):
            raise ValueError("api_bases and api_keys must have the same length")
        if max_parallel_per_endpoint < 1:
            raise ValueError("max_parallel_per_endpoint must be at least 1")
        self._endpoints = list(zip(api_bases, api_keys))
        self._available = queue.Queue()
        for endpoint in self._endpoints:
            for _ in range(max_parallel_per_endpoint):
                self._available.put(endpoint)

    def __len__(self):
        return len(self._endpoints)

    @contextmanager
    def lease(self):
        endpoint = self._available.get()
        try:
            yield endpoint
        finally:
            self._available.put(endpoint)


def parse_json_object(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def parse_selected_indices(text: str, window_len: int, count: int):
    data = parse_json_object(text)
    values = data.get("indices") if isinstance(data, dict) else None
    if not isinstance(values, list):
        values = [int(value) for value in re.findall(r"\d+", text)]

    selected = []
    for value in values:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= window_len and idx not in selected:
            selected.append(idx)
        if len(selected) >= count:
            break
    return selected


def parse_merge_decision(text: str):
    data = parse_json_object(text)
    if not isinstance(data, dict):
        return False, 2

    merge = data.get("merge", False)
    if isinstance(merge, str):
        merge = merge.strip().lower() in {"true", "yes", "y", "1"}
    else:
        merge = bool(merge)

    keep = data.get("keep", 2)
    if isinstance(keep, str):
        keep = keep.strip().lower()
        if keep in {"first", "frame1", "1", "a"}:
            keep = 1
        elif keep in {"second", "frame2", "2", "b"}:
            keep = 2
    try:
        keep = int(keep)
    except (TypeError, ValueError):
        keep = 2
    if keep not in {1, 2}:
        keep = 2
    return merge, keep


def parse_segment_tree_decision(text: str, child_count: int, child_summary_sizes=None):
    data = parse_json_object(text)
    normalized_text = re.sub(r"[*`_]", "", text)

    representative = None
    expand = None
    if isinstance(data, dict):
        representative = data.get("representative")
        if representative is None:
            representative = data.get("index")
        expand = data.get("expand")

    if isinstance(representative, list):
        representative_values = []
        for value in representative:
            try:
                representative_values.append(int(value))
            except (TypeError, ValueError):
                continue
        representative = representative_values[0] if representative_values else None

    if representative is None:
        explicit_match = re.search(r"(?i)\b(?:representative|index)\b\s*[:=]\s*(\d+)", normalized_text)
        if explicit_match:
            representative = int(explicit_match.group(1))
    if representative is None:
        for value in re.findall(r"\d+", normalized_text):
            idx = int(value)
            if 1 <= idx <= child_count:
                representative = idx
                break
    try:
        representative = int(representative)
    except (TypeError, ValueError):
        raise ValueError(f"Failed to parse representative child index from selector response: {text}")
    if representative == 0 and child_count >= 1:
        representative = 1
    if representative < 1 or representative > child_count:
        for value in re.findall(r"\d+", normalized_text):
            idx = int(value)
            if 1 <= idx <= child_count:
                representative = idx
                break
    if (
        (representative < 1 or representative > child_count)
        and child_summary_sizes
        and sum(child_summary_sizes) > 0
    ):
        running = 0
        for child_idx, size in enumerate(child_summary_sizes, start=1):
            running += size
            if representative <= running:
                representative = child_idx
                break
    if representative < 1 or representative > child_count:
        raise ValueError(f"Representative index {representative} outside 1..{child_count}")

    if isinstance(expand, str):
        expand = expand.strip().lower() in {"true", "yes", "y", "1"}
    elif expand is None:
        lower = normalized_text.lower()
        if '"expand": true' in lower or "expand=true" in lower:
            expand = True
        elif '"expand": false' in lower or "expand=false" in lower:
            expand = False
        else:
            expand_match = re.search(
                r"(?i)\bexpand\b\s*[:=]?\s*(true|false|yes|no|y|n|1|0)",
                normalized_text,
            )
            if expand_match:
                expand = expand_match.group(1).strip().lower() in {"true", "yes", "y", "1"}
    if expand is None:
        lower = normalized_text.lower()
        if "expand true" in lower:
            expand = True
        elif "expand false" in lower:
            expand = False
        else:
            raise ValueError(f"Failed to parse expand flag from selector response: {text}")
    else:
        expand = bool(expand)
    return representative, expand


def parse_arg_list(text: str | None):
    if text is None:
        return []
    return [item.strip() for item in re.split(r"[\s,]+", text) if item.strip()]


def dedupe_frame_candidates(candidates):
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate.frame_idx in seen:
            continue
        seen.add(candidate.frame_idx)
        unique.append(candidate)
    return tuple(unique)


def configure_api_endpoints(args):
    small_api_bases = parse_arg_list(args.small_api_bases) or [args.api_base]
    small_api_keys = parse_arg_list(args.small_api_keys) or [args.api_key]
    if len(small_api_keys) == 1 and len(small_api_bases) > 1:
        small_api_keys = small_api_keys * len(small_api_bases)
    if len(small_api_bases) != len(small_api_keys):
        raise ValueError("--small_api_keys must have length 1 or match --small_api_bases")

    args.small_endpoint_pool = EndpointPool(
        small_api_bases,
        small_api_keys,
        max_parallel_per_endpoint=args.small_endpoint_parallelism,
    )
    args.large_api_base = args.large_api_base or args.api_base
    args.large_api_key = args.large_api_key if args.large_api_key is not None else args.api_key


def chat_with_small_model(args, video_frames, prompt_text: str, max_tokens: int):
    with args.small_endpoint_pool.lease() as (api_base, api_key):
        return chat_with_frames(
            api_base=api_base,
            api_key=api_key,
            model_name=args.small_model_name,
            video_frames=video_frames,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            temperature=args.temperature,
            response_format={"type": "json_object"},
        )


def infer_with_large_model(args, video_frames, question: str, candidates):
    return infer_one(
        api_base=args.large_api_base,
        api_key=args.large_api_key,
        model_name=args.large_model_name,
        video_frames=video_frames,
        question=question,
        candidates=candidates,
        max_tokens=args.answer_max_tokens,
        temperature=args.temperature,
    )


def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}")
    return total_frames, fps


def load_sampled_frames(video_dir: str, video_name: str, seconds_per_frame: float):
    video_path, start, end = resolve_video_source(video_dir, video_name)
    total_frames, fps = get_video_meta(video_path)
    if start is None or end is None:
        start = 0.0
        end = total_frames / fps

    start = max(0.0, float(start))
    end = min(float(end), total_frames / fps)
    if end <= start:
        raise ValueError(f"Empty time range for {video_name}: {start}-{end}")

    times = []
    current = start
    while current < end:
        times.append(current)
        current += seconds_per_frame
    if not times:
        times = [start]

    cap = cv2.VideoCapture(video_path)
    frames = []
    seen = set()
    for time_sec in times:
        frame_idx = max(0, min(int(round(time_sec * fps)), total_frames - 1))
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(FrameCandidate(frame_idx=frame_idx, time_sec=frame_idx / fps, image=Image.fromarray(frame)))
    cap.release()
    if not frames:
        raise ValueError(f"No sampled frames loaded for {video_name}")
    return frames


def build_change_prompt(window_len: int, selected_per_window: int):
    return (
        "You are selecting informative video frames for downstream video question answering.\n"
        f"The input is an ordered sliding window of {window_len} frames sampled every 0.5 seconds.\n"
        f"Choose exactly {selected_per_window} frame numbers where the visual/action state changes the most "
        "relative to nearby frames. Prefer frames that introduce new objects, actions, poses, or scene state.\n"
        f"Return JSON only: {{\"indices\": [<1-{window_len}>, <1-{window_len}>]}}"
    )


def build_merge_prompt():
    return (
        "You are merging candidate key frames for video question answering.\n"
        "The input has two frames in chronological order: frame 1 then frame 2.\n"
        "If they show mostly the same visual/action information, set merge=true and keep the single frame with "
        "more information. If they contain distinct information, set merge=false.\n"
        "Return JSON only: {\"merge\": true/false, \"keep\": 1/2}"
    )


def format_candidates(candidates):
    if not candidates:
        return ""
    return "\n".join(
        f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates)
    )


def build_question_aware_guard_prompt(child_count: int, question: str, candidates):
    prompt = (
        "You are building a question-aware segment-tree summary for video question answering.\n"
        "Do not answer the question. Do not guess the option.\n"
        "Your job is to choose one representative frame for the interval and decide whether the interval must stay expanded.\n"
        "Avoid overcompression: if compressing to one frame may lose answer-relevant temporal or identity evidence, expand.\n"
        "Avoid undercompression: if the child frames are effectively redundant for this question, do not expand.\n"
        f"The input contains {child_count} representative frames for consecutive sub-intervals in chronological order.\n"
        f"Question: {question}\n"
    )
    if candidates:
        prompt += f"Options:\n{format_candidates(candidates)}\n"
    prompt += (
        "Set expand=true when different children may matter because of before/after order, distinct action stages, different object/person identity cues, "
        "or answer-relevant state changes.\n"
        "Set expand=false when the child frames are near-duplicates or only add continuation without new evidence for this question.\n"
        f"Return JSON only: {{\"representative\": <1-{child_count}>, \"expand\": true/false}}"
    )
    return prompt


def build_evidence_plan_prompt(question: str, candidates, router_mode: str, plan_style: str):
    prompt = (
        "You are planning what visibly grounded evidence must be preserved for downstream video question answering.\n"
        "Do not answer the question. Do not predict which option is correct. Do not rank the options.\n"
        "Do not use world knowledge, typical scripts, or hidden assumptions about what probably happened.\n"
        "Only include evidence that could be visually confirmed from video frames.\n"
        "Your job is to produce a recall-oriented evidence plan: which observable cues, entities, and temporal stages "
        "must be kept so a later model can still answer the question reliably.\n"
        f"Question: {question}\n"
    )
    if router_mode == "question_and_options" and candidates:
        prompt += (
            "Options:\n"
            f"{format_candidates(candidates)}\n"
            "Use the options only to widen the evidence checklist. Preserve evidence that could support or rule out "
            "any option. Do not state which option seems likely. Do not invent hidden events that are not visually grounded.\n"
        )
    if plan_style == "soft_temporal_chain":
        prompt += (
            "Think in terms of a soft temporal chain rather than a single decisive frame.\n"
            "Prefer short stage labels that describe what must be visually preserved before, during, and after the anchor event.\n"
        )
    prompt += (
        "Return JSON only with this schema:\n"
        "{"
        "\"question_focus\": string, "
        "\"anchor\": string, "
        "\"relations\": [string], "
        "\"stages\": [string], "
        "\"preserve\": string, "
        "\"uncertainty\": string"
        "}\n"
        "Rules:\n"
        "- anchor should name the visible event or state the later model may need to localize. If there is no clear anchor, use an empty string.\n"
        "- relations must be a JSON list with 0 to 3 unique short items chosen only from: before, after, while, same object, state change.\n"
        "- Do not repeat relation items.\n"
        "- stages must be a JSON list with 0 to 3 short stage labels in chronological order. Each item must be under 8 words.\n"
        "- preserve should be one short grounded sentence, under 30 words.\n"
        "- uncertainty should be one short grounded sentence, under 20 words.\n"
        "- If something is not visually confirmable, omit it.\n"
        "Keep everything concise and grounded."
    )
    return prompt


def normalize_relations(values):
    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = re.sub(r"\s+", " ", value.strip().lower())
        if not text:
            continue
        if text not in ALLOWED_EVIDENCE_RELATIONS:
            continue
        if text in seen:
            continue
        normalized.append(text)
        seen.add(text)
        if len(normalized) >= MAX_EVIDENCE_RELATIONS:
            break
    return normalized


def normalize_stage_labels(values):
    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = re.sub(r"\s+", " ", value.strip())
        if not text:
            continue
        lower = text.lower()
        if lower in seen:
            continue
        normalized.append(text)
        seen.add(lower)
        if len(normalized) >= MAX_EVIDENCE_STAGES:
            break
    return normalized


def parse_string_list(value):
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
    return normalize_relations(items)


def parse_stage_list(value):
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
    return normalize_stage_labels(items)


def extract_named_text(text: str, key: str):
    json_match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"\n]*)"', text, re.I)
    if json_match:
        return json_match.group(1).strip()

    truncated_json_match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^\n]*)', text, re.I)
    if truncated_json_match:
        value = truncated_json_match.group(1).strip().rstrip('",')
        if value:
            return value

    label = key.replace("_", " ")
    line_match = re.search(
        rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*[:=]\s*(.+)$",
        text,
    )
    if line_match:
        return line_match.group(1).strip()

    return ""


def extract_relations(text: str):
    json_match = re.search(r'"relations"\s*:\s*\[(.*?)\]', text, re.I | re.S)
    if json_match:
        values = re.findall(r'"([^"]+)"', json_match.group(1))
        if values:
            return normalize_relations(values)
        parts = [item.strip(" '\"\n\t") for item in json_match.group(1).split(",")]
        return normalize_relations(parts)

    line_match = re.search(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?relations(?:\*\*)?\s*[:=]\s*(.+)$", text)
    if not line_match:
        return []

    values = [item.strip(" '\"\n\t") for item in re.split(r"[;,]", line_match.group(1))]
    return normalize_relations(values)


def extract_stages(text: str):
    json_match = re.search(r'"stages"\s*:\s*\[(.*?)\]', text, re.I | re.S)
    if json_match:
        values = re.findall(r'"([^"]+)"', json_match.group(1))
        if values:
            return normalize_stage_labels(values)
        parts = [item.strip(" '\"\n\t") for item in json_match.group(1).split(",")]
        return normalize_stage_labels(parts)

    line_match = re.search(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?stages(?:\*\*)?\s*[:=]\s*(.+)$", text)
    if not line_match:
        return []

    values = [item.strip(" '\"\n\t") for item in re.split(r"[;,]", line_match.group(1))]
    return normalize_stage_labels(values)


def synthesize_preserve(question_focus: str, anchor: str, relations, stages):
    focus = anchor or question_focus
    if not focus:
        return ""
    if stages:
        stage_text = " -> ".join(stages)
        return f"Preserve the visible chain around {focus}: {stage_text}."
    if relations:
        relation_text = ", ".join(relations)
        return f"Preserve visible evidence for {focus}, including the relevant {relation_text} relation."
    return f"Preserve visible evidence for {focus} across the needed action stages and object interactions."


def clamp_text(value: str, max_chars: int):
    value = re.sub(r"\s+", " ", value.strip())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def parse_evidence_plan(text: str):
    data = parse_json_object(text)
    if isinstance(data, dict):
        question_focus = data.get("question_focus")
        if not isinstance(question_focus, str) or not question_focus.strip():
            raise ValueError(f"Evidence plan missing non-empty question_focus: {text}")
        anchor = data.get("anchor", "")
        if not isinstance(anchor, str):
            raise ValueError(f"Evidence plan anchor must be a string: {text}")
        preserve = data.get("preserve")
        if not isinstance(preserve, str):
            preserve = ""
        uncertainty = data.get("uncertainty", "")
        if not isinstance(uncertainty, str):
            raise ValueError(f"Evidence plan uncertainty must be a string: {text}")

        relations = parse_string_list(data.get("relations"))
        stages = parse_stage_list(data.get("stages"))
        preserve = preserve.strip() or synthesize_preserve(
            question_focus.strip(),
            anchor.strip(),
            relations,
            stages,
        )
        if not preserve:
            raise ValueError(f"Evidence plan missing non-empty preserve note: {text}")
        plan = {
            "question_focus": question_focus.strip(),
            "anchor": anchor.strip(),
            "relations": relations,
            "stages": stages,
            "preserve": preserve,
            "uncertainty": uncertainty.strip(),
        }
    else:
        plan = {
            "question_focus": extract_named_text(text, "question_focus"),
            "anchor": extract_named_text(text, "anchor"),
            "relations": extract_relations(text),
            "stages": extract_stages(text),
            "preserve": extract_named_text(text, "preserve"),
            "uncertainty": extract_named_text(text, "uncertainty"),
        }
        if plan["question_focus"] and not plan["preserve"]:
            plan["preserve"] = synthesize_preserve(
                plan["question_focus"],
                plan["anchor"],
                plan["relations"],
                plan["stages"],
            )
        if not plan["question_focus"] or not plan["preserve"]:
            raise ValueError(f"Failed to parse evidence plan from router response: {text}")

    plan["question_focus"] = clamp_text(plan["question_focus"], 160)
    plan["anchor"] = clamp_text(plan["anchor"], 160)
    plan["preserve"] = clamp_text(plan["preserve"], 240)
    plan["uncertainty"] = clamp_text(plan["uncertainty"], 160)
    if not any(plan[key] for key in ("anchor", "relations", "preserve", "uncertainty")):
        raise ValueError(f"Evidence plan has no usable checklist items: {text}")
    return plan


def format_evidence_plan(plan):
    if not plan:
        return ""
    return json.dumps(plan, ensure_ascii=False)


def build_segment_tree_prompt(
    child_count: int,
    evidence_plan=None,
    question: str = "",
    candidates=None,
    prompt_mode: str = "evidence_plan",
    summary_mode: str = "representative_only",
    summary_frame_cap: int = 1,
):
    if prompt_mode == "question_aware_guard":
        return build_question_aware_guard_prompt(child_count, question, candidates)
    parts = [
        "You are building a segment-tree-style hierarchical summary for a video.\n",
        "Your goal is not aggressive compression. Your goal is to preserve grounded video evidence for downstream "
        "reasoning.\n",
        f"The input contains {child_count} consecutive child intervals in chronological order.\n",
    ]
    if summary_mode == "boundary_representative":
        parts.append(
            f"Each child interval contributes up to {summary_frame_cap} summary frames in order: start, representative, end. "
            "The summaries are concatenated child by child.\n"
        )
    else:
        parts.append("Each child interval contributes one representative frame.\n")
    if evidence_plan:
        parts.append(f"Evidence plan: {format_evidence_plan(evidence_plan)}\n")
    parts.extend(
        [
            "Choose the single child frame that best represents the whole interval.\n",
            "Then decide whether a downstream model would lose any potentially relevant evidence if it only saw that "
            "one representative frame instead of all child frames.\n",
        ]
    )
    if evidence_plan:
        parts.append(
            "Use the evidence plan as a grounded recall note. Expand whenever different child frames may carry "
            "different evidence stages, different identity cues, different temporal relations, or different ambiguity-resolving details.\n"
        )
        stages = evidence_plan.get("stages") if isinstance(evidence_plan, dict) else None
        if stages:
            parts.append(
                f"Preserve the visible temporal chain across stages: {' -> '.join(stages)}. "
                "If one child interval does not cover the whole chain, prefer expand=true.\n"
            )
    parts.extend(
        [
            "Set expand=false only when the child frames are near-duplicates and preserve the same objects, action "
            "stage, state changes, temporal relations, and scene evidence.\n",
            "Never rely on common sense or likely unseen actions. If the needed evidence is not visibly grounded in "
            "the representative frame, set expand=true.\n",
            "Set expand=true if any child frame adds non-redundant information about object identity, before/after "
            "order, hand-object interaction, scene state, or causal progression. If uncertain, set expand=true.\n",
            f"Return JSON only: {{\"representative\": <1-{child_count}>, \"expand\": true/false}}",
        ]
    )
    return "".join(parts)


def choose_representative_frame(frames, anchor_time: float | None):
    if not frames:
        raise ValueError("frames must not be empty")
    if anchor_time is None:
        return frames[len(frames) // 2]
    return min(frames, key=lambda frame: (abs(frame.time_sec - anchor_time), frame.frame_idx))


def summarize_node_frames(frames, representative: FrameCandidate, summary_mode: str, summary_frame_cap: int):
    if summary_mode == "representative_only" or summary_frame_cap <= 1:
        return (representative,)
    candidates = [frames[0], representative, frames[-1]]
    summary = list(dedupe_frame_candidates(candidates))
    return tuple(summary[:summary_frame_cap])


def summarize_internal_node(children, representative: FrameCandidate, summary_mode: str, summary_frame_cap: int):
    if summary_mode == "representative_only" or summary_frame_cap <= 1:
        return (representative,)
    first = children[0].summary_frames[0]
    last = children[-1].summary_frames[-1]
    summary = list(dedupe_frame_candidates([first, representative, last]))
    return tuple(summary[:summary_frame_cap])


def flatten_child_summary_frames(children):
    frames = []
    for child in children:
        frames.extend(child.summary_frames)
    return frames


def build_keyframe_window_leaves(sampled_frames, video_dir: str, video_name: str, question_id: str, args):
    if not args.key_frame_data:
        return None
    video_path, start, end = resolve_video_time_range(video_dir, video_name)
    fps = get_video_fps(video_path)
    clip_start_frame = int(start * fps)
    clip_end_frame = max(clip_start_frame + 1, int(end * fps))
    keyframe_indices = load_question_keyframes(args.key_frame_data, question_id)
    if not keyframe_indices:
        return None

    filtered = [idx for idx in keyframe_indices if clip_start_frame <= idx < clip_end_frame]
    if not filtered:
        return None
    if args.max_keyframe_anchors > 0:
        filtered = filtered[: args.max_keyframe_anchors]
    anchor_times = [idx / fps for idx in filtered]
    windows = build_anchor_windows(start, end, anchor_times)
    groups = [[] for _ in windows]

    window_idx = 0
    for sample_idx, frame in enumerate(sampled_frames):
        while window_idx + 1 < len(windows) and frame.time_sec >= windows[window_idx][1]:
            window_idx += 1
        groups[window_idx].append(sample_idx)

    leaves = []
    for anchor_time, window, group in zip(anchor_times, windows, groups):
        if not group:
            continue
        segment_frames = [sampled_frames[idx] for idx in group]
        representative = choose_representative_frame(segment_frames, anchor_time)
        summary_frames = summarize_node_frames(
            segment_frames,
            representative,
            args.segment_tree_summary_mode,
            args.segment_tree_summary_frame_cap,
        )
        leaves.append(
            SegmentTreeNode(
                node_id=f"leaf_{len(leaves)}",
                level=0,
                start_idx=group[0],
                end_idx=group[-1],
                start_time=window[0],
                end_time=window[1],
                representative=representative,
                summary_frames=summary_frames,
                children=(),
                expand=False,
            )
        )
    return leaves or None


def build_leaf_nodes(sampled_frames, video_dir: str, video_name: str, question_id: str, args):
    if args.segment_tree_leaf_source == "keyframe_windows":
        leaves = build_keyframe_window_leaves(sampled_frames, video_dir, video_name, question_id, args)
        if leaves:
            return leaves

    nodes = []
    for idx, frame in enumerate(sampled_frames):
        summary_frames = summarize_node_frames(
            [frame],
            frame,
            args.segment_tree_summary_mode,
            args.segment_tree_summary_frame_cap,
        )
        nodes.append(
            SegmentTreeNode(
                node_id=f"leaf_{idx}",
                level=0,
                start_idx=idx,
                end_idx=idx,
                start_time=frame.time_sec,
                end_time=frame.time_sec,
                representative=frame,
                summary_frames=summary_frames,
                children=(),
                expand=False,
            )
        )
    return nodes


def select_window_frames(frames, args):
    if len(frames) <= args.selected_per_window:
        return frames

    selected_by_idx = {}
    for start in range(0, len(frames), args.window_stride):
        window = frames[start : start + args.window_size]
        if len(window) < 2:
            continue
        if len(window) <= args.selected_per_window:
            chosen = list(range(1, len(window) + 1))
        else:
            response = chat_with_small_model(
                args,
                video_frames=[item.image for item in window],
                prompt_text=build_change_prompt(len(window), args.selected_per_window),
                max_tokens=args.selector_max_tokens,
            )
            chosen = parse_selected_indices(response, len(window), args.selected_per_window)
        for local_idx in chosen:
            candidate = window[local_idx - 1]
            selected_by_idx.setdefault(candidate.frame_idx, candidate)

    if not selected_by_idx:
        return [frames[0]]
    return [selected_by_idx[idx] for idx in sorted(selected_by_idx)]


def merge_adjacent_candidates(candidates, args):
    if len(candidates) <= 1:
        return candidates

    kept = [candidates[0]]
    for candidate in candidates[1:]:
        previous = kept[-1]
        response = chat_with_small_model(
            args,
            video_frames=[previous.image, candidate.image],
            prompt_text=build_merge_prompt(),
            max_tokens=args.selector_max_tokens,
        )
        merge, keep = parse_merge_decision(response)
        if merge:
            if keep == 2:
                kept[-1] = candidate
        else:
            kept.append(candidate)
    return kept


def cap_final_frames(frames, max_frames: int):
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (max_frames - 1)
    indices = [round(i * step) for i in range(max_frames)]
    return [frames[idx] for idx in indices]


def prepare_final_images(final_frames, resize_to: int):
    images = [item.image for item in final_frames]
    if resize_to <= 0:
        return images
    return [image.resize((resize_to, resize_to), Image.BICUBIC) for image in images]


def load_final_images(video_dir: str, video_name: str, frame_indices, resize_to: int):
    if not frame_indices:
        raise ValueError(f"No final frame indices available for {video_name}")
    video_path, _, _ = resolve_video_source(video_dir, video_name)
    images, _ = load_video_frame_indices(video_path, frame_indices)
    if not images:
        raise ValueError(f"Failed to reload final frames for {video_name}")
    if resize_to <= 0:
        return images
    return [image.resize((resize_to, resize_to), Image.BICUBIC) for image in images]


def build_segment_tree(sampled_frames, evidence_plan, question: str, candidates, question_id: str, video_dir: str, video_name: str, args):
    nodes = build_leaf_nodes(sampled_frames, video_dir, video_name, question_id, args)
    if len(nodes) == 1:
        return nodes[0]

    current = nodes
    level = 1
    internal_idx = 0
    while len(current) > 1:
        next_level = []
        for start in range(0, len(current), args.tree_branching):
            children = current[start : start + args.tree_branching]
            if len(children) == 1:
                next_level.append(children[0])
                continue

            response = chat_with_small_model(
                args,
                video_frames=[frame.image for frame in flatten_child_summary_frames(children)],
                prompt_text=build_segment_tree_prompt(
                    len(children),
                    evidence_plan=evidence_plan,
                    question=question,
                    candidates=candidates,
                    prompt_mode=args.segment_tree_prompt_mode,
                    summary_mode=args.segment_tree_summary_mode,
                    summary_frame_cap=args.segment_tree_summary_frame_cap,
                ),
                max_tokens=args.selector_max_tokens,
            )
            representative_idx, expand = parse_segment_tree_decision(
                response,
                len(children),
                child_summary_sizes=[len(child.summary_frames) for child in children],
            )
            representative = children[representative_idx - 1].representative
            summary_frames = summarize_internal_node(
                children,
                representative,
                args.segment_tree_summary_mode,
                args.segment_tree_summary_frame_cap,
            )
            parent = SegmentTreeNode(
                node_id=f"node_{level}_{internal_idx}",
                level=level,
                start_idx=children[0].start_idx,
                end_idx=children[-1].end_idx,
                start_time=children[0].start_time,
                end_time=children[-1].end_time,
                representative=representative,
                summary_frames=summary_frames,
                children=tuple(children),
                expand=expand,
            )
            next_level.append(parent)
            internal_idx += 1
        current = next_level
        level += 1
    return current[0]


def collect_segment_tree_frontier(node, selected):
    if node.is_leaf or not requires_segment_tree_expansion(node):
        selected.append(node)
        return
    for child in node.children:
        collect_segment_tree_frontier(child, selected)


def requires_segment_tree_expansion(node):
    if node.is_leaf:
        return False
    if node.end_idx - node.start_idx + 1 > MAX_FRONTIER_LEAVES:
        return True
    if node.expand:
        return True
    return any(requires_segment_tree_expansion(child) for child in node.children)


def serialize_segment_tree(root):
    ordered = []
    visited = set()

    def visit(node):
        if node.node_id in visited:
            return
        visited.add(node.node_id)
        ordered.append(
            {
                "node_id": node.node_id,
                "level": node.level,
                "sample_span": [node.start_idx, node.end_idx],
                "time_span": [round(node.start_time, 3), round(node.end_time, 3)],
                "representative_frame_idx": node.representative.frame_idx,
                "representative_time": round(node.representative.time_sec, 3),
                "summary_frame_indices": [frame.frame_idx for frame in node.summary_frames],
                "summary_times": [round(frame.time_sec, 3) for frame in node.summary_frames],
                "expand": node.expand,
                "expand_subtree": requires_segment_tree_expansion(node),
                "children": [child.node_id for child in node.children],
            }
        )
        for child in node.children:
            visit(child)

    visit(root)
    return ordered


def collect_frontier_sampled_frames(sampled, selected_nodes):
    final_frames = []
    seen = set()
    for node in selected_nodes:
        for frame in sampled[node.start_idx : node.end_idx + 1]:
            if frame.frame_idx in seen:
                continue
            seen.add(frame.frame_idx)
            final_frames.append(frame)
    return final_frames


def collect_frontier_summary_frames(selected_nodes):
    final_frames = []
    seen = set()
    for node in selected_nodes:
        for frame in node.summary_frames:
            if frame.frame_idx in seen:
                continue
            seen.add(frame.frame_idx)
            final_frames.append(frame)
    return final_frames


def build_evidence_plan(question: str, candidates, args):
    if args.segment_tree_prompt_mode == "question_aware_guard":
        return None
    if args.segment_tree_router_mode == "none":
        return None
    response = chat_with_small_model(
        args,
        video_frames=[],
        prompt_text=build_evidence_plan_prompt(
            question,
            candidates,
            args.segment_tree_router_mode,
            args.segment_tree_plan_style,
        ),
        max_tokens=args.evidence_plan_max_tokens,
    )
    return parse_evidence_plan(response)


def select_segment_tree_frames(video_dir: str, video_name: str, question: str, candidates, question_id: str, args):
    sampled = load_sampled_frames(video_dir, video_name, args.seconds_per_frame)
    evidence_plan = build_evidence_plan(question, candidates, args)
    root = build_segment_tree(
        sampled,
        evidence_plan,
        question,
        candidates,
        question_id,
        video_dir,
        video_name,
        args,
    )
    selected_nodes = []
    collect_segment_tree_frontier(root, selected_nodes)
    selected_nodes.sort(key=lambda node: node.start_idx)
    if args.segment_tree_payload_mode == "sampled_in_frontier":
        final_frames = collect_frontier_sampled_frames(sampled, selected_nodes)
    elif args.segment_tree_payload_mode == "node_summary":
        final_frames = collect_frontier_summary_frames(selected_nodes)
    else:
        final_frames = [node.representative for node in selected_nodes]
    final_frames = cap_final_frames(final_frames, args.max_final_frames)
    return sampled, root, selected_nodes, final_frames, evidence_plan


def select_swarm_frames(video_dir: str, video_name: str, question: str, candidates, question_id: str, args):
    if args.selection_mode == "segment_tree":
        sampled, root, selected_nodes, final_frames, evidence_plan = select_segment_tree_frames(
            video_dir,
            video_name,
            question,
            candidates,
            question_id,
            args,
        )
        tree_nodes = serialize_segment_tree(root)
        return {
            "sampled": sampled,
            "final_frames": final_frames,
            "swarm": {
                "selection_mode": "segment_tree",
                "prompt_mode": args.segment_tree_prompt_mode,
                "plan_style": args.segment_tree_plan_style,
                "router_mode": args.segment_tree_router_mode,
                "payload_mode": args.segment_tree_payload_mode,
                "leaf_source": args.segment_tree_leaf_source,
                "summary_mode": args.segment_tree_summary_mode,
                "final_image_size": args.final_image_size,
                "sampled_count": len(sampled),
                "tree_branching": args.tree_branching,
                "tree_levels": max(node["level"] for node in tree_nodes) + 1,
                "node_count": len(tree_nodes),
                "frontier_count": len(selected_nodes),
                "selected_node_ids": [node.node_id for node in selected_nodes],
                "selected_spans": [
                    [round(node.start_time, 3), round(node.end_time, 3)] for node in selected_nodes
                ],
                "final_count": len(final_frames),
                "final_frame_indices": [item.frame_idx for item in final_frames],
                "final_times": [round(item.time_sec, 3) for item in final_frames],
                "evidence_plan": evidence_plan,
                "tree_nodes": tree_nodes,
            },
        }

    sampled = load_sampled_frames(video_dir, video_name, args.seconds_per_frame)
    window_selected = select_window_frames(sampled, args)
    merged = merge_adjacent_candidates(window_selected, args)
    final_frames = cap_final_frames(merged, args.max_final_frames)
    return {
        "sampled": sampled,
        "final_frames": final_frames,
        "swarm": {
            "selection_mode": "window_merge",
            "final_image_size": args.final_image_size,
            "sampled_count": len(sampled),
            "window_selected_count": len(window_selected),
            "merged_count": len(merged),
            "final_count": len(final_frames),
            "final_frame_indices": [item.frame_idx for item in final_frames],
            "final_times": [round(item.time_sec, 3) for item in final_frames],
        },
    }


def process_sample(sample, args):
    selection = select_swarm_frames(
        args.video_dir,
        sample["video_name"],
        sample["question"],
        sample["candidates"],
        sample["question_id"],
        args,
    )
    final_images = prepare_final_images(selection["final_frames"], args.final_image_size)
    pred = infer_with_large_model(args, final_images, sample["question"], sample["candidates"])
    return {
        "task_name": sample["task_name"],
        "question": sample["question"],
        "id": sample["question_id"],
        "answer_number": int(sample["answer_number"]),
        "candidates": sample["candidates"],
        "answer": sample["answer"],
        "pred": pred,
        "swarm": selection["swarm"],
    }


def route_sample(sample, args):
    selection = select_swarm_frames(
        args.video_dir,
        sample["video_name"],
        sample["question"],
        sample["candidates"],
        sample["question_id"],
        args,
    )
    return {
        "sample": sample,
        "swarm": selection["swarm"],
    }


def answer_routed_sample(routed, args):
    sample = routed["sample"]
    final_images = load_final_images(
        args.video_dir,
        sample["video_name"],
        routed["swarm"]["final_frame_indices"],
        args.final_image_size,
    )
    pred = infer_with_large_model(args, final_images, sample["question"], sample["candidates"])
    return {
        "task_name": sample["task_name"],
        "question": sample["question"],
        "id": sample["question_id"],
        "answer_number": int(sample["answer_number"]),
        "candidates": sample["candidates"],
        "answer": sample["answer"],
        "pred": pred,
        "swarm": routed["swarm"],
    }


def main(args):
    if args.selection_mode == "segment_tree" and args.tree_branching < 2:
        raise ValueError("--tree_branching must be at least 2 for segment_tree mode")
    configure_api_endpoints(args)
    route_workers = args.route_workers if args.route_workers > 0 else args.workers
    if route_workers <= 0:
        route_workers = len(args.small_endpoint_pool)
    answer_workers = max(1, args.answer_workers)

    with open(args.gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    error_path = os.path.join(args.output_dir, f"{args.output_name}.errors.log")

    done_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
    pending = [sample for sample in data if sample["question_id"] not in done_ids]

    lock = threading.Lock()
    with open(output_path, "a", encoding="utf-8") as out_f, open(error_path, "a", encoding="utf-8") as err_f:
        with ThreadPoolExecutor(max_workers=route_workers) as route_executor, ThreadPoolExecutor(
            max_workers=answer_workers
        ) as answer_executor:
            route_futures = {
                route_executor.submit(route_sample, sample, args): sample for sample in pending
            }
            answer_futures = {}
            progress = tqdm(total=len(pending), desc="Swarm frame eval")
            while route_futures or answer_futures:
                active = list(route_futures) + list(answer_futures)
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in route_futures:
                        sample = route_futures.pop(future)
                        try:
                            routed = future.result()
                        except Exception as exc:
                            with lock:
                                err_f.write(f"{sample['question_id']}: {type(exc).__name__}: {exc}\n")
                                err_f.flush()
                            progress.update(1)
                            continue
                        answer_future = answer_executor.submit(answer_routed_sample, routed, args)
                        answer_futures[answer_future] = sample
                        continue

                    sample = answer_futures.pop(future)
                    try:
                        record = future.result()
                        with lock:
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_f.flush()
                    except Exception as exc:
                        with lock:
                            err_f.write(f"{sample['question_id']}: {type(exc).__name__}: {exc}\n")
                            err_f.flush()
                    finally:
                        progress.update(1)
            progress.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--api_base", default="http://10.126.62.90:8003/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--small_api_bases", default=None)
    parser.add_argument("--small_api_keys", default=None)
    parser.add_argument("--large_api_base", default=None)
    parser.add_argument("--large_api_key", default=None)
    parser.add_argument("--small_model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--large_model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--key_frame_path", default=None)
    parser.add_argument("--selection_mode", choices=["window_merge", "segment_tree"], default="window_merge")
    parser.add_argument("--seconds_per_frame", type=float, default=0.5)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=5)
    parser.add_argument("--selected_per_window", type=int, default=2)
    parser.add_argument("--tree_branching", type=int, default=2)
    parser.add_argument(
        "--segment_tree_leaf_source",
        choices=["sampled_frames", "keyframe_windows"],
        default="sampled_frames",
    )
    parser.add_argument(
        "--segment_tree_router_mode",
        choices=["none", "question_only", "question_and_options"],
        default="question_and_options",
    )
    parser.add_argument(
        "--segment_tree_payload_mode",
        choices=["representative", "node_summary", "sampled_in_frontier"],
        default="representative",
    )
    parser.add_argument(
        "--segment_tree_prompt_mode",
        choices=["evidence_plan", "question_aware_guard"],
        default="evidence_plan",
    )
    parser.add_argument(
        "--segment_tree_plan_style",
        choices=["grounded", "soft_temporal_chain"],
        default="soft_temporal_chain",
    )
    parser.add_argument(
        "--segment_tree_summary_mode",
        choices=["representative_only", "boundary_representative"],
        default="boundary_representative",
    )
    parser.add_argument("--segment_tree_summary_frame_cap", type=int, default=3)
    parser.add_argument("--max_keyframe_anchors", type=int, default=0)
    parser.add_argument("--final_image_size", type=int, default=0)
    parser.add_argument("--max_final_frames", type=int, default=0)
    parser.add_argument("--selector_max_tokens", type=int, default=96)
    parser.add_argument("--evidence_plan_max_tokens", type=int, default=512)
    parser.add_argument("--answer_max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--route_workers", type=int, default=0)
    parser.add_argument("--answer_workers", type=int, default=1)
    parser.add_argument("--small_endpoint_parallelism", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.key_frame_data = None
    if args.key_frame_path:
        with open(args.key_frame_path, "r", encoding="utf-8") as f:
            args.key_frame_data = json.load(f)
    main(args)
