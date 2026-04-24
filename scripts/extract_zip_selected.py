import argparse
import os
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("name_list")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    needed = {line.strip() for line in Path(args.name_list).read_text().splitlines() if line.strip()}
    os.makedirs(args.output_dir, exist_ok=True)

    extracted = 0
    with ZipFile(args.zip_path) as zf:
        for member in zf.namelist():
            base = os.path.basename(member)
            if base in needed:
                target_path = os.path.join(args.output_dir, base)
                if os.path.exists(target_path):
                    extracted += 1
                    continue
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted += 1
                print(f"extracted {extracted}: {base}")

    print(f"done: {extracted}")


if __name__ == "__main__":
    main()
