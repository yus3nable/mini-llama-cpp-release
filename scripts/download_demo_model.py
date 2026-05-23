#!/usr/bin/env python3
"""
Download a small GGUF chat model for demo purposes.

Default: Qwen2-0.5B-Instruct Q8_0 from bartowski's HuggingFace repository.
"""

import argparse
import os
import sys
import urllib.request
import json

DEFAULT_REPO = "bartowski/Qwen2-0.5B-Instruct-GGUF"
DEFAULT_FILE = "Qwen2-0.5B-Instruct-Q8_0.gguf"
DEFAULT_DIR = "models/chat"


def get_hf_download_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download_file(url: str, dest: str):
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def main():
    parser = argparse.ArgumentParser(description="Download demo GGUF chat model")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace repo")
    parser.add_argument("--file", default=DEFAULT_FILE, help="GGUF filename")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Local download directory")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    dest = os.path.join(args.dir, args.file)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"Model already exists: {dest} ({size_mb:.1f} MB)")
        return 0

    url = get_hf_download_url(args.repo, args.file)
    try:
        download_file(url, dest)
    except Exception as e:
        print(f"Error downloading: {e}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"Downloaded: {dest} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
