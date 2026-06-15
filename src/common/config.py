"""
Single source of truth for filesystem paths — all derived RELATIVE to this
file's location (no hardcoded drive letters), so the repo is portable and
survives being moved / junctioned to another drive.

    src/common/config.py  ->  parents[2] == project root
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT     = PROJECT_ROOT / "src"
DATA_ROOT    = PROJECT_ROOT / "data"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# Per-dataset OUTPUT roots (results CSVs + plots).
FSC_OUTPUTS     = OUTPUTS_ROOT / "fsc"
MINTREC_OUTPUTS = OUTPUTS_ROOT / "mintrec"
IEMOCAP_OUTPUTS = OUTPUTS_ROOT / "iemocap"

# Per-dataset DATA roots. FSC artifacts currently live at the data/ top level
# (s3prl___superb, fsc_small_ablation, teacher_features, teacher_probe, student);
# new datasets get their own subdir.
MINTREC_DATA = DATA_ROOT / "mintrec"

# IEMOCAP: place the official release at  data/iemocap/IEMOCAP_full_release/
# (data/ is junctioned to E:). The corpus is licence-gated — request access
# from USC SAIL; it cannot be auto-downloaded.
IEMOCAP_DATA = DATA_ROOT / "iemocap"


def ensure_src_on_path():
    """Put SRC_ROOT on sys.path so `import common...` / `import fsc...` resolve
    regardless of how a script is launched. Idempotent."""
    import sys
    s = str(SRC_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)
