"""
Gradio mic demo for the tiny FSC student.

Sanity check only, not an evaluation. The vocabulary is the closed 31-intent
FSC set, so anything outside it gets forced into the nearest one. Live mic
accuracy comes out lower than the test numbers because the speaker, mic and
room are all different from the training data.

The preprocessing is imported rather than rewritten, so it can't drift from
what the model was trained on:
  wav_to_logmel from fsc.student.precompute_logmel
  train mean/std from data/student/logmel_cache/train_logmel.pt
  DSResNetSE with SMALL_KW from fsc.student.final_test

Any of the 8 final-test checkpoints can be picked and compared on the same
recording (Qwen teacher x 4 methods, HuBERT teacher x 4 methods).

    python src/fsc/live_demo/app.py

then open the printed localhost url and allow the mic. localhost counts as a
secure context so there's no https to set up.
"""

import json
import sys
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import DATA_ROOT                                   # noqa: E402
from common.models.audio_student import DSResNetSE                    # noqa: E402
from fsc.student.precompute_logmel import wav_to_logmel, SR, LABEL_CFG  # noqa: E402
from fsc.student.final_test import SMALL_KW                           # noqa: E402
from fsc.student.kd_common import LOGMEL, DEVICE                      # noqa: E402


QWEN_DIR = DATA_ROOT / "student" / "final_test_checkpoints"
HUBERT_DIR = DATA_ROOT / "student" / "hubert_final_test_checkpoints"
CHECKPOINTS = {
    "Qwen (multimodal) - Full KD":      QWEN_DIR / "full_kd.pt",
    "Qwen (multimodal) - Logit KD":     QWEN_DIR / "logit_kd.pt",
    "Qwen (multimodal) - Feature KD":   QWEN_DIR / "feature_kd.pt",
    "Qwen (multimodal) - CE-only":      QWEN_DIR / "ce_only.pt",
    "HuBERT (audio-only) - Full KD":    HUBERT_DIR / "full_kd.pt",
    "HuBERT (audio-only) - Logit KD":   HUBERT_DIR / "logit_kd.pt",
    "HuBERT (audio-only) - Feature KD": HUBERT_DIR / "feature_kd.pt",
    "HuBERT (audio-only) - CE-only":    HUBERT_DIR / "ce_only.pt",
}
DEFAULT_CHECKPOINT = "Qwen (multimodal) - Full KD"


with open(LABEL_CFG, encoding="utf-8") as f:
    LABEL2ID = json.load(f)["label2id"]
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
N_CLASSES = len(ID2LABEL)


_train_cache = torch.load(LOGMEL / "train_logmel.pt", weights_only=False)
MEAN = _train_cache["mean"].clone().view(1, 1, 1, -1)
STD = _train_cache["std"].clone().view(1, 1, 1, -1)
del _train_cache


def _load_model(ckpt_path):
    model = DSResNetSE(**SMALL_KW).to(DEVICE)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


print(f"Device: {DEVICE}")
print("Loading 8 student checkpoints...")
MODELS = {name: _load_model(path) for name, path in CHECKPOINTS.items()}
print("Ready.")


def _to_16k_float(sr, wav):
    wav = np.asarray(wav)
    if wav.ndim > 1:                                    # to mono
        wav = wav.mean(axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        wav = wav.astype(np.float32)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


@torch.no_grad()
def predict(audio, checkpoint_name):
    if audio is None:
        return None
    sr, wav = audio
    wav = _to_16k_float(sr, wav)

    logmel = wav_to_logmel(wav)                                      # [T, 64]
    X = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).float()   # [1, 1, T, 64]
    X = ((X - MEAN) / STD).to(DEVICE)

    model = MODELS[checkpoint_name]
    _, logits = model(X)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {ID2LABEL[i]: float(probs[i]) for i in range(N_CLASSES)}


EXAMPLE_COMMANDS = [
    "Turn on the kitchen lights", "Turn off the lamp", "Turn the lights on",
    "Resume", "Turn off the music", "Turn the volume down", "Volume up",
    "Turn up the temperature in the bedroom", "Turn the kitchen temperature down",
    "Bring me my shoes", "Get me the newspaper", "Set language to Chinese",
]

with gr.Blocks(title="FSC tiny-student live demo") as demo:
    gr.Markdown(
        "# FSC tiny audio-only student -- live microphone demo\n"
        "**Qualitative sanity check only -- not a formal evaluation.** "
        "The model only knows **31 fixed smart-home intents** from the FSC "
        "dataset; speak something close to one of these (full list in "
        "`data/fsc_small_ablation/config.json`), e.g.:\n\n"
        + "\n".join(f"- {c}" for c in EXAMPLE_COMMANDS) + "\n\n"
        "Record once, then switch the checkpoint dropdown to compare how "
        "different teachers/methods classify the **same** recording."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], type="numpy", label="Speak a command")
        ckpt_dd = gr.Dropdown(choices=list(CHECKPOINTS.keys()), value=DEFAULT_CHECKPOINT,
                               label="Student checkpoint (97,991 params, ~0.37 MB FP32)")
    predict_btn = gr.Button("Predict intent", variant="primary")
    label_out = gr.Label(num_top_classes=5, label="Predicted intent (top 5)")

    predict_btn.click(predict, inputs=[audio_in, ckpt_dd], outputs=label_out)
    audio_in.change(predict, inputs=[audio_in, ckpt_dd], outputs=label_out)
    ckpt_dd.change(predict, inputs=[audio_in, ckpt_dd], outputs=label_out)


if __name__ == "__main__":
    demo.launch()
