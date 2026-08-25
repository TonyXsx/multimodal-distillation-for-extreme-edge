"""
All paths live here. Everything is relative to this file, so no drive letters
end up hardcoded and the repo survives being moved to another disk.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT     = PROJECT_ROOT / "src"
DATA_ROOT    = PROJECT_ROOT / "data"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# results csvs + plots, one root per dataset
FSC_OUTPUTS     = OUTPUTS_ROOT / "fsc"
MINTREC_OUTPUTS = OUTPUTS_ROOT / "mintrec"

# FSC stuff still sits at the top of data/ for historical reasons; anything new
# gets its own subdir.
MINTREC_DATA = DATA_ROOT / "mintrec"


def ensure_src_on_path():
    """put src/ on sys.path so `import common...` works however the script was
    launched. safe to call twice."""
    import sys
    s = str(SRC_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)
