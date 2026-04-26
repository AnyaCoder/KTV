import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--feature_output", required=True)
    parser.add_argument("--keyframe_output", required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--dinov2_repo", default=".codex_tmp/dinov2")
    parser.add_argument("--weights_path", default="dinov2_vitl14_reg4_pretrain.pth")
    parser.add_argument("--max_frames", type=int, default=5400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_clusters", type=int, default=6)
    parser.add_argument(
        "--ranking_mode",
        choices=["question_only", "question_candidates"],
        default="question_only",
    )
    parser.add_argument("--clip_model", default="ViT-L/14")
    parser.add_argument("--dino_flush_every", type=int, default=8)
    parser.add_argument("--keyframe_flush_every", type=int, default=16)
    parser.add_argument("--skip_dino", action="store_true")
    parser.add_argument("--skip_keyframe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_devices(spec: str):
    devices = [part.strip() for part in spec.split(",") if part.strip()]
    if not devices:
        raise ValueError("devices must not be empty")
    return devices


def shard_output_path(path: str, shard_rank: int):
    target = Path(path)
    return str(target.with_name(f"{target.stem}.shard{shard_rank}{target.suffix}"))


def prepare_device(token: str):
    if token.isdigit():
        return {"visible": token, "device_arg": "cuda:0"}
    if token.startswith("cuda:") and token[5:].isdigit():
        return {"visible": token[5:], "device_arg": "cuda:0"}
    return {"visible": None, "device_arg": token}


def launch_subprocesses(stage_name: str, commands):
    procs = []
    for item in commands:
        print(f"[{stage_name}] launch shard={item['shard_rank']} device={item['device_label']}")
        sys.stdout.flush()
        proc = subprocess.Popen(
            item["cmd"],
            cwd=item["cwd"],
            env=item["env"],
        )
        procs.append((item, proc))

    failed = []
    for item, proc in procs:
        code = proc.wait()
        if code != 0:
            failed.append((item["shard_rank"], code))
    if failed:
        raise RuntimeError(f"{stage_name} failed for shards: {failed}")


def merge_pickles(shard_paths, output_path: str):
    merged = {}
    for path in shard_paths:
        with open(path, "rb") as f:
            part = pickle.load(f)
        overlap = set(merged).intersection(part)
        if overlap:
            raise ValueError(f"Duplicate feature keys across shards: {sorted(overlap)[:5]}")
        merged.update(part)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(merged, f)
    os.replace(tmp_path, output_path)


def merge_json_files(shard_paths, output_path: str):
    merged = {}
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as f:
            part = json.load(f)
        overlap = set(merged).intersection(part)
        if overlap:
            raise ValueError(f"Duplicate json keys across shards: {sorted(overlap)[:5]}")
        merged.update(part)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def main():
    args = parse_args()
    if args.skip_dino and args.skip_keyframe:
        raise ValueError("skip_dino and skip_keyframe cannot both be set")

    repo_root = Path(__file__).resolve().parents[1]
    devices = parse_devices(args.devices)
    num_shards = len(devices)
    python_bin = sys.executable

    keyframe_script = repo_root / "keyframe_select_new.py"
    cluster_script = repo_root / "cluster_keyframe_and_order.py"

    feature_shards = [shard_output_path(args.feature_output, rank) for rank in range(num_shards)]
    if not args.skip_dino:
        dino_commands = []
        for rank, token in enumerate(devices):
            prepared = prepare_device(token)
            env = os.environ.copy()
            if prepared["visible"] is not None:
                env["CUDA_VISIBLE_DEVICES"] = prepared["visible"]
            cmd = [
                python_bin,
                str(keyframe_script),
                "--gt_file",
                args.gt_file,
                "--video_dir",
                args.video_dir,
                "--output_pickle",
                feature_shards[rank],
                "--dinov2_repo",
                args.dinov2_repo,
                "--weights_path",
                args.weights_path,
                "--device",
                prepared["device_arg"],
                "--max_frames",
                str(args.max_frames),
                "--batch_size",
                str(args.batch_size),
                "--num_shards",
                str(num_shards),
                "--shard_rank",
                str(rank),
                "--flush_every",
                str(args.dino_flush_every),
            ]
            if args.resume:
                cmd.append("--resume")
            dino_commands.append(
                {
                    "cmd": cmd,
                    "cwd": str(repo_root),
                    "env": env,
                    "shard_rank": rank,
                    "device_label": token,
                }
            )

        launch_subprocesses("dino", dino_commands)
        merge_pickles(feature_shards, args.feature_output)
        print(f"[merge] wrote merged features to {args.feature_output}")
        sys.stdout.flush()
    elif not Path(args.feature_output).exists():
        raise FileNotFoundError(f"feature_output does not exist: {args.feature_output}")

    keyframe_shards = [shard_output_path(args.keyframe_output, rank) for rank in range(num_shards)]
    if not args.skip_keyframe:
        cluster_commands = []
        for rank, token in enumerate(devices):
            prepared = prepare_device(token)
            env = os.environ.copy()
            if prepared["visible"] is not None:
                env["CUDA_VISIBLE_DEVICES"] = prepared["visible"]
            cmd = [
                python_bin,
                str(cluster_script),
                "--gt_file",
                args.gt_file,
                "--feature_pickle",
                args.feature_output,
                "--output_json",
                keyframe_shards[rank],
                "--num_clusters",
                str(args.num_clusters),
                "--ranking_mode",
                args.ranking_mode,
                "--clip_model",
                args.clip_model,
                "--device",
                prepared["device_arg"],
                "--num_shards",
                str(num_shards),
                "--shard_rank",
                str(rank),
                "--flush_every",
                str(args.keyframe_flush_every),
            ]
            if args.resume:
                cmd.append("--resume")
            cluster_commands.append(
                {
                    "cmd": cmd,
                    "cwd": str(repo_root),
                    "env": env,
                    "shard_rank": rank,
                    "device_label": token,
                }
            )

        launch_subprocesses("keyframe", cluster_commands)
        merge_json_files(keyframe_shards, args.keyframe_output)
        print(f"[merge] wrote merged keyframes to {args.keyframe_output}")


if __name__ == "__main__":
    main()
