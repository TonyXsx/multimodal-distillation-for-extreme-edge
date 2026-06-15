"""
Download MIntRec2.0 raw data (annotations + packed videos) into data/mintrec/.

Source: HuggingFace THUIAR/MMLA-Datasets -> MIntRec2.0/  (public, no token needed;
set HF_TOKEN only for faster rate limits). ~9 GB video tarball. Idempotent: HF
resumes interrupted downloads, and tar extraction is skipped once .mp4 are present.

Lays files out as (data/ is junctioned to E:):

    data/mintrec/MIntRec2.0/
        train.tsv  dev.tsv  test.tsv
        MIntRec2.0_video.tar.gz
        video/ ... *.mp4         (extracted clips, named MIntRec2.0_{dia}_{utt}.mp4)

Usage:
    python src/mintrec/teacher_probe/download_data.py
    python src/mintrec/teacher_probe/download_data.py --no-extract   # fetch tar only
"""

import argparse
import sys
import tarfile
from pathlib import Path

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

HF_REPO   = "THUIAR/MMLA-Datasets"
ANNO_DIR  = MINTREC_DATA / "MIntRec2.0"
VIDEO_DIR = ANNO_DIR / "video"
FILES     = ["train.tsv", "dev.tsv", "test.tsv", "MIntRec2.0_video.tar.gz"]


def main():
    ap = argparse.ArgumentParser(description="Download MIntRec2.0 from THUIAR/MMLA-Datasets.")
    ap.add_argument("--no-extract", action="store_true", help="download the tarball but skip extraction")
    args = ap.parse_args()

    MINTREC_DATA.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        print(f"Fetching {fn} ...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset",
            filename=f"MIntRec2.0/{fn}", local_dir=str(MINTREC_DATA),
        )

    if args.no_extract:
        print("Skipping extraction (--no-extract).")
        return

    if VIDEO_DIR.exists() and any(VIDEO_DIR.rglob("*.mp4")):
        print(f"Videos already extracted at {VIDEO_DIR} -- skipping.")
    else:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        tar_path = ANNO_DIR / "MIntRec2.0_video.tar.gz"
        print(f"Extracting {tar_path.name} (~9 GB, be patient) ...", flush=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(VIDEO_DIR)
        n = sum(1 for _ in VIDEO_DIR.rglob("*.mp4"))
        print(f"Extraction done: {n} .mp4 clips under {VIDEO_DIR}")


if __name__ == "__main__":
    main()
