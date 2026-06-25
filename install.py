"""Download and unpack Radio-Flux-Solver catalogs. Run with: uv run install.py"""

import os
import tarfile
import requests
from tqdm import tqdm

REPO_ID = "EV4g/Radio-Flux-Solver-Catalogs"
FILENAME = "catalogs.tar.gz"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FILENAME}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOGS_DIR = os.path.join(SCRIPT_DIR, "catalogs")
MARKER = os.path.join(CATALOGS_DIR, ".downloaded")
ARCHIVE_PATH = os.path.join(CATALOGS_DIR, FILENAME)

def download_with_progress(url, dest):
    existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    response = requests.get(url, headers=headers, stream=True)

    if existing_size and response.status_code == 416:
        print("File already fully downloaded.")
        return

    total_size = existing_size + int(response.headers.get("content-length", 0))
    mode = "ab" if existing_size else "wb"

    with open(dest, mode) as f, tqdm(
        desc=FILENAME, total=total_size, initial=existing_size,
        unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

def main():
    os.makedirs(CATALOGS_DIR, exist_ok=True)

    if os.path.exists(MARKER):
        print("Catalogs already present, skipping download.")
        return

    print(f"Downloading {FILENAME}...")
    download_with_progress(URL, ARCHIVE_PATH)

    print("\nUnpacking into catalogs/...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=SCRIPT_DIR)

    os.remove(ARCHIVE_PATH)
    print("Removed archive.")

    open(MARKER, "w").close()
    print("Catalogs ready in ./catalogs/")

if __name__ == "__main__":
    main()
