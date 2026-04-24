import argparse
import math
import os
import threading

import requests


def get_size(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    size = response.headers.get("Content-Length")
    if size is None:
        raise ValueError("Missing Content-Length header.")
    return int(size)


def download_range(url: str, part_path: str, start: int, end: int):
    existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    current_start = start + existing
    if current_start > end:
        return

    headers = {"Range": f"bytes={current_start}-{end}"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        mode = "ab" if existing else "wb"
        with open(part_path, mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def merge_parts(output_path: str, part_paths):
    with open(output_path, "wb") as out_f:
        for part_path in part_paths:
            with open(part_path, "rb") as in_f:
                while True:
                    chunk = in_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--parts", type=int, default=8)
    args = parser.parse_args()

    total_size = get_size(args.url)
    part_size = math.ceil(total_size / args.parts)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    part_paths = [f"{args.output}.part{i:02d}" for i in range(args.parts)]
    threads = []
    for i in range(args.parts):
        start = i * part_size
        end = min(total_size - 1, (i + 1) * part_size - 1)
        t = threading.Thread(
            target=download_range, args=(args.url, part_paths[i], start, end), daemon=False
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    merge_parts(args.output, part_paths)
    for part_path in part_paths:
        if os.path.exists(part_path):
            os.remove(part_path)

    print(args.output)


if __name__ == "__main__":
    main()
