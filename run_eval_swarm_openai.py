import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
from PIL import Image
from tqdm import tqdm

from run_inference_openai_compatible import chat_with_frames, infer_one, resolve_video_source

MAX_FRONTIER_LEAVES = 4


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
    children: tuple["SegmentTreeNode", ...]
    expand: bool

    @property
    def is_leaf(self) -> bool:
        return not self.children


class EndpointPool:
    def __init__(self, api_bases, api_keys):
        if not api_bases:
            raise ValueError("api_bases must not be empty")
        if len(api_bases) != len(api_keys):
            raise ValueError("api_bases and api_keys must have the same length")
        self._endpoints = list(zip(api_bases, api_keys))
        self._next_idx = 0
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            endpoint = self._endpoints[self._next_idx]
            self._next_idx = (self._next_idx + 1) % len(self._endpoints)
        return endpoint


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


def parse_segment_tree_decision(text: str, child_count: int):
    data = parse_json_object(text)

    representative = None
    expand = None
    if isinstance(data, dict):
        representative = data.get("representative")
        if representative is None:
            representative = data.get("index")
        expand = data.get("expand")

    if representative is None:
        for value in re.findall(r"\d+", text):
            idx = int(value)
            if 1 <= idx <= child_count:
                representative = idx
                break
    try:
        representative = int(representative)
    except (TypeError, ValueError):
        raise ValueError(f"Failed to parse representative child index from selector response: {text}")
    if representative < 1 or representative > child_count:
        raise ValueError(f"Representative index {representative} outside 1..{child_count}")

    if isinstance(expand, str):
        expand = expand.strip().lower() in {"true", "yes", "y", "1"}
    elif expand is None:
        lower = text.lower()
        if '"expand": true' in lower or "expand=true" in lower:
            expand = True
        elif '"expand": false' in lower or "expand=false" in lower:
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


def configure_api_endpoints(args):
    small_api_bases = parse_arg_list(args.small_api_bases) or [args.api_base]
    small_api_keys = parse_arg_list(args.small_api_keys) or [args.api_key]
    if len(small_api_keys) == 1 and len(small_api_bases) > 1:
        small_api_keys = small_api_keys * len(small_api_bases)
    if len(small_api_bases) != len(small_api_keys):
        raise ValueError("--small_api_keys must have length 1 or match --small_api_bases")

    args.small_endpoint_pool = EndpointPool(small_api_bases, small_api_keys)
    args.large_api_base = args.large_api_base or args.api_base
    args.large_api_key = args.large_api_key if args.large_api_key is not None else args.api_key


def chat_with_small_model(args, video_frames, prompt_text: str, max_tokens: int):
    api_base, api_key = args.small_endpoint_pool.next()
    return chat_with_frames(
        api_base=api_base,
        api_key=api_key,
        model_name=args.small_model_name,
        video_frames=video_frames,
        prompt_text=prompt_text,
        max_tokens=max_tokens,
        temperature=args.temperature,
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


def build_evidence_plan_prompt(question: str, candidates, router_mode: str):
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
        options = "\n".join(
            f"({chr(ord('A') + idx)}) {candidate}" for idx, candidate in enumerate(candidates)
        )
        prompt += (
            "Options:\n"
            f"{options}\n"
            "Use the options only to widen the evidence checklist. Preserve evidence that could support or rule out "
            "any option. Do not state which option seems likely. Do not invent hidden events that are not visually grounded.\n"
        )
    prompt += (
        "Return JSON only with this schema:\n"
        "{"
        "\"question_focus\": string, "
        "\"anchor\": string, "
        "\"relations\": [string], "
        "\"preserve\": string, "
        "\"uncertainty\": string"
        "}\n"
        "Rules:\n"
        "- anchor should name the visible event or state the later model may need to localize. If there is no clear anchor, use an empty string.\n"
        "- relations should contain only short temporal or identity relations such as before, after, while, same object, state change.\n"
        "- preserve should be one short paragraph describing what kinds of visually grounded evidence or stages must not be lost.\n"
        "- uncertainty should be one short paragraph describing which ambiguities still require keeping extra evidence.\n"
        "- If something is not visually confirmable, omit it.\n"
        "Keep everything concise and grounded."
    )
    return prompt


def parse_string_list(value):
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
    return items


def parse_evidence_plan(text: str):
    data = parse_json_object(text)
    if not isinstance(data, dict):
        raise ValueError(f"Failed to parse evidence plan from router response: {text}")

    question_focus = data.get("question_focus")
    if not isinstance(question_focus, str) or not question_focus.strip():
        raise ValueError(f"Evidence plan missing non-empty question_focus: {text}")
    anchor = data.get("anchor", "")
    if not isinstance(anchor, str):
        raise ValueError(f"Evidence plan anchor must be a string: {text}")
    preserve = data.get("preserve")
    if not isinstance(preserve, str) or not preserve.strip():
        raise ValueError(f"Evidence plan missing non-empty preserve note: {text}")
    uncertainty = data.get("uncertainty", "")
    if not isinstance(uncertainty, str):
        raise ValueError(f"Evidence plan uncertainty must be a string: {text}")

    plan = {
        "question_focus": question_focus.strip(),
        "anchor": anchor.strip(),
        "relations": parse_string_list(data.get("relations")),
        "preserve": preserve.strip(),
        "uncertainty": uncertainty.strip(),
    }
    if not any(plan[key] for key in ("anchor", "relations", "preserve", "uncertainty")):
        raise ValueError(f"Evidence plan has no usable checklist items: {text}")
    return plan


def format_evidence_plan(plan):
    if not plan:
        return ""
    return json.dumps(plan, ensure_ascii=False)


def build_segment_tree_prompt(child_count: int, evidence_plan=None):
    parts = [
        "You are building a segment-tree-style hierarchical summary for a video.\n",
        "Your goal is not aggressive compression. Your goal is to preserve grounded video evidence for downstream "
        "reasoning.\n",
        f"The input contains {child_count} representative frames for consecutive sub-intervals in chronological order.\n",
    ]
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


def build_segment_tree(sampled_frames, evidence_plan, args):
    nodes = [
        SegmentTreeNode(
            node_id=f"leaf_{idx}",
            level=0,
            start_idx=idx,
            end_idx=idx,
            start_time=frame.time_sec,
            end_time=frame.time_sec,
            representative=frame,
            children=(),
            expand=False,
        )
        for idx, frame in enumerate(sampled_frames)
    ]
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
                video_frames=[child.representative.image for child in children],
                prompt_text=build_segment_tree_prompt(len(children), evidence_plan),
                max_tokens=args.selector_max_tokens,
            )
            representative_idx, expand = parse_segment_tree_decision(response, len(children))
            representative = children[representative_idx - 1].representative
            parent = SegmentTreeNode(
                node_id=f"node_{level}_{internal_idx}",
                level=level,
                start_idx=children[0].start_idx,
                end_idx=children[-1].end_idx,
                start_time=children[0].start_time,
                end_time=children[-1].end_time,
                representative=representative,
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


def build_evidence_plan(question: str, candidates, args):
    if args.segment_tree_router_mode == "none":
        return None
    response = chat_with_small_model(
        args,
        video_frames=[],
        prompt_text=build_evidence_plan_prompt(question, candidates, args.segment_tree_router_mode),
        max_tokens=args.evidence_plan_max_tokens,
    )
    return parse_evidence_plan(response)


def select_segment_tree_frames(video_dir: str, video_name: str, question: str, candidates, args):
    sampled = load_sampled_frames(video_dir, video_name, args.seconds_per_frame)
    evidence_plan = build_evidence_plan(question, candidates, args)
    root = build_segment_tree(sampled, evidence_plan, args)
    selected_nodes = []
    collect_segment_tree_frontier(root, selected_nodes)
    selected_nodes.sort(key=lambda node: node.start_idx)
    if args.segment_tree_payload_mode == "sampled_in_frontier":
        final_frames = collect_frontier_sampled_frames(sampled, selected_nodes)
    else:
        final_frames = [node.representative for node in selected_nodes]
    final_frames = cap_final_frames(final_frames, args.max_final_frames)
    return sampled, root, selected_nodes, final_frames, evidence_plan


def select_swarm_frames(video_dir: str, video_name: str, question: str, candidates, args):
    if args.selection_mode == "segment_tree":
        sampled, root, selected_nodes, final_frames, evidence_plan = select_segment_tree_frames(
            video_dir,
            video_name,
            question,
            candidates,
            args,
        )
        tree_nodes = serialize_segment_tree(root)
        return {
            "sampled": sampled,
            "final_frames": final_frames,
            "swarm": {
                "selection_mode": "segment_tree",
                "router_mode": args.segment_tree_router_mode,
                "payload_mode": args.segment_tree_payload_mode,
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


def main(args):
    if args.selection_mode == "segment_tree" and args.tree_branching < 2:
        raise ValueError("--tree_branching must be at least 2 for segment_tree mode")
    configure_api_endpoints(args)

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
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, sample, args): sample for sample in pending}
            progress = tqdm(total=len(futures), desc="Swarm frame eval")
            for future in as_completed(futures):
                sample = futures[future]
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
    parser.add_argument("--selection_mode", choices=["window_merge", "segment_tree"], default="window_merge")
    parser.add_argument("--seconds_per_frame", type=float, default=0.5)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--window_stride", type=int, default=5)
    parser.add_argument("--selected_per_window", type=int, default=2)
    parser.add_argument("--tree_branching", type=int, default=2)
    parser.add_argument(
        "--segment_tree_router_mode",
        choices=["none", "question_only", "question_and_options"],
        default="question_and_options",
    )
    parser.add_argument(
        "--segment_tree_payload_mode",
        choices=["representative", "sampled_in_frontier"],
        default="representative",
    )
    parser.add_argument("--final_image_size", type=int, default=0)
    parser.add_argument("--max_final_frames", type=int, default=0)
    parser.add_argument("--selector_max_tokens", type=int, default=96)
    parser.add_argument("--evidence_plan_max_tokens", type=int, default=256)
    parser.add_argument("--answer_max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
