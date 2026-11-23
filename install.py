"""Download and unpack Radio-Flux-Solver catalogs. Run with: uv run install.py"""

import argparse
import os
import tarfile
import requests
from tqdm import tqdm

REPO_ID = "EV4g/Radio-Flux-Solver-Catalogs"
FILENAME = "catalogs.tar.gz"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FILENAME}"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def _unpack(archive_path, dest):
    """Unpack the catalogs archive into dest, stripping the leading catalogs/ so
    the survey folders (vlssr/, tgss/, ...) land directly inside."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("catalogs/")]
        for m in members:
            m.name = m.name[len("catalogs/"):]
        tar.extractall(path=dest, members=members, filter="data")

def main():
    parser = argparse.ArgumentParser(description="Download and unpack the Radio-Flux-Solver reference catalogs.")
    parser.add_argument("--path", default=None, help="Directory to unpack the catalogs into (default: this folder's catalogs/). The survey folders land directly inside.")
    args = parser.parse_args()

    catalogs_dir = os.path.abspath(args.path) if args.path else os.path.join(SCRIPT_DIR, "catalogs")
    marker      = os.path.join(catalogs_dir, ".downloaded")
    archive_path = os.path.join(catalogs_dir, FILENAME)

    os.makedirs(catalogs_dir, exist_ok=True)

    if os.path.exists(marker):
        print("Catalogs already present, skipping download.")
        return

    print(f"Downloading {FILENAME}...")
    download_with_progress(URL, archive_path)

    print(f"\nUnpacking into {catalogs_dir}...")
    _unpack(archive_path, catalogs_dir)

    os.remove(archive_path)
    print("Removed archive.")

    open(marker, "w").close()
    print(f"Catalogs ready in {catalogs_dir}")

if __name__ == "__main__":
    main()
