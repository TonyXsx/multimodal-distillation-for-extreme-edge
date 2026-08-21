"""
IEMOCAP data layer for the teacher stage.

Frozen extraction, the QLoRA fine-tune and adapted extraction all read
utterances through this module, so the prompt wording, the input ordering and
the label mapping exist in exactly one place and cannot drift between the
training run and the extraction that has to reproduce it.

INPUT ORDER -- instruction -> audio -> transcript -> "Emotion:" readout.

Audio sits immediately after the instruction so that, under causal masking,
audio tokens attend only to the task instruction and never to the transcript.
That keeps `audio_mean` a genuinely audio-only feature (something a student
could in principle reproduce) while the readout token at the end still sees
everything, so a single forward pass yields both a clean audio feature and a
transcript-aware readout.

This is the arrangement validated on MIntRec2.0. Note that the *frozen*
MIntRec extractor used the opposite order (audio last, absorbing the preceding
text); here both the frozen and the adapted arm use this same order, so the
frozen-vs-adapted control is a like-for-like comparison rather than a
comparison that also changes the input layout.

No video: only Session 1 ships any, at dialog level, and Session 1 is the test
split. The student on this track is audio-only.
"""

import sys
from pathlib import Path

import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_EXTRACTED, IEMOCAP_MANIFEST  # noqa: E402
from iemocap.data.build_manifest import CLASSES, LABEL2ID      # noqa: E402  single source of truth
# librosa 16 kHz mono loader, reused rather than re-implemented.
from mintrec.teacher_probe.extract_features_local import load_audio  # noqa: E402,F401

SPLITS = ("train", "val", "test")

INSTRUCTION = (
    "You are analyzing a short clip of conversational speech to recognize the "
    "speaker's emotion, among four classes: angry, happy, neutral, sad. Attend "
    "to HOW it is said -- the tone of voice and prosody -- not only the words."
)
READOUT_CUE = "Emotion:"


def load_split(split, limit=None, manifest=None):
    """One split of the manifest, ordered deterministically.

    Row order fixes the shard order, so it must not depend on filesystem
    listing: `build_manifest.py` already sorts by (session, dialog, turn_id)
    and that order is preserved here.
    """
    path = Path(manifest) if manifest else IEMOCAP_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run src/iemocap/data/build_manifest.py first")
    df = pd.read_csv(path, keep_default_na=False, dtype={"transcript": str})
    if split != "all":
        df = df[df["split"] == split]
    df = df.reset_index(drop=True)
    return df if limit is None else df.iloc[:limit].reset_index(drop=True)


def wav_path(row):
    return IEMOCAP_EXTRACTED / row["wav_path"]


def build_inputs(proc, device, wav, transcript, use_transcript=True):
    """instruction -> audio -> [transcript] -> readout cue.

    An empty transcript is simply omitted: 29 utterances in the 4-class subset
    consist of nothing but a stripped non-verbal marker, and they keep their
    audio and label.
    """
    from qwen_omni_utils import process_mm_info

    content = [{"type": "text", "text": INSTRUCTION},
               {"type": "audio", "audio": wav}]
    if use_transcript and isinstance(transcript, str) and transcript.strip():
        content.append({"type": "text", "text": f'Transcript: "{transcript.strip()}"'})
    content.append({"type": "text", "text": READOUT_CUE})

    conv = [{"role": "user", "content": content}]
    txt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    a, i, v = process_mm_info(conv, use_audio_in_video=False)
    return proc(text=txt, audio=a, images=i, videos=v, return_tensors="pt",
                padding=True, use_audio_in_video=False).to(device)


def input_order_str(use_transcript=True):
    mid = " -> transcript" if use_transcript else ""
    return f"instruction -> audio{mid} -> '{READOUT_CUE}' (readout last)"


__all__ = ["CLASSES", "LABEL2ID", "SPLITS", "INSTRUCTION", "READOUT_CUE",
           "load_split", "wav_path", "load_audio", "build_inputs", "input_order_str"]
