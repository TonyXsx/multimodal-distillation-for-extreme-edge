"""
IEMOCAP paths, built on the shared roots in common.config.

Kept in this package rather than added to common/config.py so the IEMOCAP track
stays self-contained and nothing existing had to be edited.

Two protocols sit side by side, picked with the IEMOCAP_PROTOCOL env var so one
export switches the whole pipeline:

    si (default)  speaker-independent, train S2,3,4 / val S5 / test S1. The
                  main protocol, no speaker shared between splits.
    sd            speaker-dependent, one stratified 60/20/20 draw over all
                  5,531 utterances on speaker x emotion, so every speaker and
                  class shows up in all three splits.

sd is a control, not a result. It leaks speakers on purpose, to measure what
speaker-independence costs and to see whether distillation helps once that
difficulty is gone. Its numbers don't compare with any speaker-independent
figure, including our own si ones and including the published IEMOCAP results
that use it (MS-SENet and TIM-Net both do plain shuffled KFold over utterances,
so theirs are speaker-dependent too).

Artefacts get a per-protocol suffix so the two never overwrite each other.
"""

import os
import sys
from pathlib import Path

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT, OUTPUTS_ROOT  # noqa: E402

#   si        sessions 2,3,4 / 5 / 1, the original speaker-independent split
#   sd        stratified on speaker x emotion, all 10 speakers in every split
#   loso1..5  leave-one-session-out, fold k holds out session k as test
_PROTOCOLS = ("si", "sd") + tuple(f"loso{k}" for k in range(1, 6))
PROTOCOL = os.environ.get("IEMOCAP_PROTOCOL", "si").lower()
if PROTOCOL not in _PROTOCOLS:
    raise ValueError(f"IEMOCAP_PROTOCOL must be one of {_PROTOCOLS}, got {PROTOCOL!r}")
_SUF = "" if PROTOCOL == "si" else f"_{PROTOCOL}"

# raw release, the five Session*.zip plus Documentation.zip
IEMOCAP_DATA = DATA_ROOT / "iemocap"

# the bit we actually unpack: labels, transcripts, utterance wavs. shared by
# both protocols, the audio doesn't care how it gets split
IEMOCAP_EXTRACTED = IEMOCAP_DATA / "extracted"

# manifest, one per protocol
IEMOCAP_MANIFEST = IEMOCAP_DATA / f"manifest{_SUF}.csv"

# teacher artefacts. the LoRA teacher has to be retrained per protocol - it
# memorises its training split (97.9 UA on data it saw vs 79.6 on unseen
# speakers), so reusing the si teacher under sd would hand the student
# near-oracle soft labels on most of sd-train
IEMOCAP_FEATURES = IEMOCAP_DATA / f"teacher_features{_SUF}"
IEMOCAP_QLORA = IEMOCAP_DATA / f"teacher_qlora{_SUF}"
IEMOCAP_PROBE = IEMOCAP_DATA / f"teacher_probe{_SUF}"

# student caches: log-mel, mfcc, augmented
IEMOCAP_STUDENT = IEMOCAP_DATA / f"student{_SUF}"

# results csvs. one tree, protocol goes in the filename
IEMOCAP_OUTPUTS = OUTPUTS_ROOT / "iemocap"


def find_adapted_features(root=None):
    """find the LoRA feature dir for whichever protocol is active.

    The dir name has the adapter epoch in it, and that epoch is picked on val
    UA so it differs between protocols. Better to look it up than hardcode a
    name that quietly points at the wrong one. Highest epoch wins.
    """
    root = root or IEMOCAP_FEATURES
    cands = sorted(d for d in root.glob("*LORA*") if d.is_dir())
    if not cands:
        raise FileNotFoundError(
            f"no LoRA-adapted feature directory under {root} "
            f"(protocol={PROTOCOL}) -- run teacher/extract_features.py --adapter ... first")
    return cands[-1]
