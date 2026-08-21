"""
IEMOCAP-specific paths, derived from the shared roots in `common.config`.

Kept local to this package (rather than added to `common/config.py`) so the
IEMOCAP track is self-contained and no existing module needs editing.

TWO PROTOCOLS live side by side, selected by the IEMOCAP_PROTOCOL environment
variable so that a single export switches the whole pipeline without touching
any script:

    si  (default)  speaker-INdependent -- train S2,3,4 / val S5 / test S1.
                   The main protocol; splits share no speaker.
    sd             speaker-dependent -- one stratified 60/20/20 draw over all
                   5,531 utterances, stratified on speaker x emotion, so every
                   speaker and every class appears in all three splits.

The `sd` protocol exists as a CONTROL, not as a result: it deliberately leaks
speakers in order to measure what speaker-independence costs, and to test
whether distillation helps once that particular difficulty is removed. Its
absolute numbers are not comparable with any speaker-independent figure --
including our own `si` numbers, and including the published IEMOCAP results
that use it (MS-SENet and TIM-Net both run plain shuffled KFold over
utterances, so their reported accuracies are speaker-dependent).

Artefacts are suffixed per protocol so the two never overwrite each other.
"""

import os
import sys
from pathlib import Path

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT, OUTPUTS_ROOT  # noqa: E402

PROTOCOL = os.environ.get("IEMOCAP_PROTOCOL", "si").lower()
if PROTOCOL not in ("si", "sd"):
    raise ValueError(f"IEMOCAP_PROTOCOL must be 'si' or 'sd', got {PROTOCOL!r}")
_SUF = "" if PROTOCOL == "si" else f"_{PROTOCOL}"

# Raw release: the five Session*.zip + Documentation.zip live here.
IEMOCAP_DATA = DATA_ROOT / "iemocap"

# Selectively extracted subset (labels + transcripts + utterance wavs).
# Shared by both protocols -- the audio does not depend on how it is split.
IEMOCAP_EXTRACTED = IEMOCAP_DATA / "extracted"

# Per-protocol utterance manifest.
IEMOCAP_MANIFEST = IEMOCAP_DATA / f"manifest{_SUF}.csv"

# Teacher-stage artefacts. The LoRA teacher must be retrained per protocol:
# it memorises its training split (97.9 UA on data it trained on against 79.6
# on unseen speakers), so reusing the si teacher under sd would hand the
# student near-oracle soft labels on the ~60% of sd-train it had already seen.
IEMOCAP_FEATURES = IEMOCAP_DATA / f"teacher_features{_SUF}"
IEMOCAP_QLORA = IEMOCAP_DATA / f"teacher_qlora{_SUF}"
IEMOCAP_PROBE = IEMOCAP_DATA / f"teacher_probe{_SUF}"

# Student-stage caches (log-mel, MFCC, augmented variants).
IEMOCAP_STUDENT = IEMOCAP_DATA / f"student{_SUF}"

# Results / stats CSVs -- one tree, protocol carried in the filenames.
IEMOCAP_OUTPUTS = OUTPUTS_ROOT / "iemocap"


def find_adapted_features(root=None):
    """Locate the LoRA-adapted feature directory for the active protocol.

    The directory name carries the adapter epoch (`..._adapter_ep3_...`), which
    is chosen by validation UA and so differs between protocols. Discovering it
    beats hard-coding a name that silently points at the wrong protocol's
    features. The highest epoch wins when several are present.
    """
    root = root or IEMOCAP_FEATURES
    cands = sorted(d for d in root.glob("*LORA*") if d.is_dir())
    if not cands:
        raise FileNotFoundError(
            f"no LoRA-adapted feature directory under {root} "
            f"(protocol={PROTOCOL}) -- run teacher/extract_features.py --adapter ... first")
    return cands[-1]
