#!/usr/bin/env python3
import os
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_file(url, target_path):
    """Attempt to download a file from 'url' to 'target_path' up to 3 tries."""
    tries = 3
    for attempt in range(tries):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 404:
                return None
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.RequestException as e:
            if attempt < tries - 1:
                continue
            else:
                raise Exception(f"Failed to download {url} after {tries} attempts") from e

def download_chunk_files(chunk_id, base_url, target_dir):
    """Download all files for a given chunk in batches until a 404 is encountered."""
    os.makedirs(target_dir, exist_ok=True)
    batch_size = 500
    file_index = 0

    while True:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {}
            for _ in range(batch_size):
                file_url = f"{base_url}/chunk{chunk_id}/example_train_{file_index}.jsonl.zst?download=true"
                target_path = os.path.join(target_dir, f"ch{chunk_id}_example_train_{file_index}.jsonl.zst")
                futures[executor.submit(download_file, file_url, target_path)] = file_index
                file_index += 1

            break_after_loop = False
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    break_after_loop = True

            if break_after_loop:
                return

def main():
    parser = argparse.ArgumentParser(
        description="Download files for specified chunks from a base URL."
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="The base directory where the datasets will be saved."
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="List of chunk IDs to download (default: 1 2 ... 10)."
    )
    args = parser.parse_args()

    base_url = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train"
    target_dir_base = args.target_dir

    for chunk_id in args.chunks:
        target_dir = os.path.join(target_dir_base, f"chunk{chunk_id}")
        print(f"Downloading chunk {chunk_id} to {target_dir}...")
        download_chunk_files(chunk_id, base_url, target_dir)

if __name__ == "__main__":
    main()