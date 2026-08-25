"""
Data layer for the IEMOCAP teacher stage.

The frozen extraction, the QLoRA fine-tune and the adapted extraction all read
utterances through here, so the prompt, the input order and the label map live
in one place and cannot drift between the training run and the extraction that
has to reproduce it.

Input order is instruction -> audio -> transcript -> "Emotion:" cue.

Audio comes right after the instruction so that under causal masking the audio
tokens see the task but never the transcript. That keeps audio_mean a genuinely
audio-only feature, something a student could actually reproduce, while the
readout token at the end still sees everything. One forward pass, both a clean
audio feature and a transcript-aware readout.

Same arrangement as MIntRec2.0, except the frozen MIntRec extractor put audio
last. Here both arms use this order, so the frozen vs adapted comparison does
not also change the layout.

No video. Only Session 1 has any, at dialog level, and Session 1 is the test
split. The student here is audio-only.
"""

import sys
from pathlib import Path

import pandas as pd

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_EXTRACTED, IEMOCAP_MANIFEST  # noqa: E402
from iemocap.data.build_manifest import CLASSES, LABEL2ID      # noqa: E402  single source of truth
# reuse the librosa 16k mono loader rather than writing another one
from mintrec.teacher_probe.extract_features_local import load_audio  # noqa: E402,F401

SPLITS = ("train", "val", "test")


def manifest_splits():
    """which splits this manifest actually has, in a fixed order.

    The LOSO folds have no val set at all, so nothing downstream should assume
    all three exist. Loops ask here rather than walking SPLITS blindly.
    """
    df = pd.read_csv(IEMOCAP_MANIFEST, usecols=["split"])
    present = set(df["split"].unique())
    return tuple(s for s in SPLITS if s in present)

INSTRUCTION = (
    "You are analyzing a short clip of conversational speech to recognize the "
    "speaker's emotion, among four classes: angry, happy, neutral, sad. Attend "
    "to HOW it is said -- the tone of voice and prosody -- not only the words."
)
READOUT_CUE = "Emotion:"


def load_split(split, limit=None, manifest=None):
    """one split of the manifest, in a deterministic order.

    Row order decides shard order, so it must not depend on directory listing.
    build_manifest.py already sorts by (session, dialog, turn_id) and that gets
    preserved here.
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

    Empty transcripts get left out. 29 utterances in the 4-class subset are
    nothing but a stripped marker, and they keep their audio and label.
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
