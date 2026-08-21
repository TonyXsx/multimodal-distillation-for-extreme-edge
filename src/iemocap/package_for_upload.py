"""
Pack the minimum IEMOCAP subset needed for teacher extraction into one tarball.

Only the utterances that survive the 4-class filter are included -- 5,531 of
10,039 -- plus the manifest itself. Everything the teacher stage never touches
(dropped classes, motion capture, forced alignments, full-dialog wavs,
Session 1's videos) stays behind, which roughly halves the upload against the
already-slimmed `extracted/` tree.

The archive unpacks directly into a DATA_ROOT, so on the remote machine:

    tar xzf iemocap_teacher_subset.tar.gz -C /root/autodl-tmp/data
    # -> data/iemocap/manifest.csv
    #    data/iemocap/extracted/Session*/sentences/wav/<dialog>/<turn>.wav

Usage:
    python src/iemocap/package_for_upload.py
    python src/iemocap/package_for_upload.py --out /tmp/iemocap.tar.gz --dry-run
"""

import argparse
import sys
import tarfile
from pathlib import Path

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_EXTRACTED, IEMOCAP_MANIFEST  # noqa: E402
from iemocap.teacher.data import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Tar the 4-class IEMOCAP subset for upload.")
    ap.add_argument("--out", default=None, help="output .tar.gz (default: alongside the data dir)")
    ap.add_argument("--dry-run", action="store_true", help="report size, write nothing")
    args = ap.parse_args()

    out = Path(args.out) if args.out else IEMOCAP_DATA / "iemocap_teacher_subset.tar.gz"
    df = load_split("all")
    wavs = [IEMOCAP_EXTRACTED / p for p in df["wav_path"]]

    missing = [p for p in wavs if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} wav files missing, e.g. {missing[:3]}")
    raw = sum(p.stat().st_size for p in wavs) + IEMOCAP_MANIFEST.stat().st_size
    print(f"{len(wavs)} wavs + manifest = {raw / 1e6:.0f} MB uncompressed")

    if args.dry_run:
        print("Dry run -- nothing written.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out} ...", flush=True)
    with tarfile.open(out, "w:gz") as tf:
        tf.add(IEMOCAP_MANIFEST, arcname="iemocap/manifest.csv")
        for i, p in enumerate(wavs, 1):
            tf.add(p, arcname=f"iemocap/extracted/{p.relative_to(IEMOCAP_EXTRACTED).as_posix()}")
            if i % 1000 == 0:
                print(f"  ... {i}/{len(wavs)}", flush=True)

    print(f"\nDone: {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("On the remote box:\n"
          "    tar xzf iemocap_teacher_subset.tar.gz -C <DATA_ROOT>")


if __name__ == "__main__":
    main()
