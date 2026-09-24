"""Rebuild catalogs.tar.gz (for install.py) from local. Run with: uv run catalogs_to_tar.py"""

import os
import shutil
import subprocess
import tarfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_PATH = os.path.join(SCRIPT_DIR, "catalogs.tar.gz")

#### catalog files to pack; paths are relative to this file and kept as-is inside the tar
CATALOG_FILES = [
    "catalogs/lolss/lolss_clean.fits",
    "catalogs/vlssr/vlssr_clean.fits",
    "catalogs/lofar/LoTSS_DR3_v1.0.srl_clean.fits",
    "catalogs/tgss/tgss_clean.fits",
    "catalogs/gleam_x_gp/gleam_x_gp_clean.fits",
    "catalogs/gleam_300/gleam_300_clean.fits",
    "catalogs/wenss/wenss_clean.fits",
    "catalogs/vcss/vcss_clean.fits",
    "catalogs/txs/txs_clean.fits",
    "catalogs/racs/racs_low_gal_clean.fits",
    "catalogs/racs/racs_low_clean.fits",
    "catalogs/apertif/apertif_clean.fits",
    "catalogs/meerkat/meerkat_clean.fits",
    "catalogs/racs/racs_mid_clean.fits",
    "catalogs/nvss/nvss_clean.fits",
    "catalogs/first/first_clean.fits",
    "catalogs/racs/racs_high_clean.fits",
    "catalogs/vlass/vlass_clean.fits",
]

def main():
    missing = [path for path in CATALOG_FILES if not os.path.exists(os.path.join(SCRIPT_DIR, path))]
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    if shutil.which("pigz"):
        # parallel via tar + pigz
        subprocess.run(["tar", "-I", "pigz", "-cvf", ARCHIVE_PATH, *CATALOG_FILES], cwd=SCRIPT_DIR, check=True)
    else:
        print("pigz not found, falling back to single-threaded gzip")
        with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
            for path in CATALOG_FILES:
                tar.add(os.path.join(SCRIPT_DIR, path), arcname=path)
                print(f"added {path}")

    size_mb = os.path.getsize(ARCHIVE_PATH) / (1024 * 1024)
    print(f"\nWrote {ARCHIVE_PATH} ({size_mb:.1f} MB, {len(CATALOG_FILES)} files)")
    print("Upload it to the HuggingFace dataset used by install.py")

if __name__ == "__main__":
    main()
