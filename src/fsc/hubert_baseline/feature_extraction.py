"""
Frozen HuBERT-large audio-only teacher: feature extraction for the FSC baseline.

Companion to src/fsc/frozen_feature_extraction (Qwen). Purpose: give the
prompted multimodal Qwen2.5-Omni teacher an audio-only control, so the KD
comparison can separate "any strong frozen teacher helps" from "the
multimodal/prompted teacher specifically helps". See README.md in this
folder for the full rationale and protocol.

Model: facebook/hubert-large-ll60k -- the plain SSL-pretrained checkpoint
(masked prediction of k-means cluster ids over Libri-Light, 60k h of
UNLABELED speech). No text or ASR fine-tuning anywhere in this checkpoint's
history, unlike e.g. the "-ls960-ft" CTC-finetuned variant, which must NOT
be used here since CTC fine-tuning against transcripts would reintroduce
text supervision and defeat the point of an audio-only control.

Feature: mean-pooled over time, final transformer layer hidden states.
No layer/prompt-order ablation is run here -- the Qwen-side ablation already
answered "which representation is best" for that teacher, and this baseline
is explicitly scoped to skip a repeat ablation; the single standard
frozen-SSL pooling recipe (as used throughout the SUPERB benchmark) is used
as-is.

Splits: train + validation only. Test is intentionally NOT extracted --
mirrors frozen_feature_extraction/full_feature_extraction.py's rationale:
the final student is audio-only and test must stay teacher-free to avoid
leakage.

Sample order / ids / labels come from fsc.student.precompute_logmel.load_fsc()
(imported, not duplicated), so sample_ids line up 1:1 with the existing
data/student/logmel_cache/*.pt used by the student KD scripts.

Output (same dict schema as the Qwen teacher_features banks):
    data/teacher_features/fsc_full__hubert-large-ll60k__last_layer_mean/
        train_features.pt
        val_features.pt
        extraction_config.json
"""

import io
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT                    # noqa: E402
from fsc.student.precompute_logmel import load_fsc      # noqa: E402  (reuse: identical labels/order/ids)

MODEL_NAME = "facebook/hubert-large-ll60k"
FEATURE_NAME = "hubert_large_ll60k_last_layer_mean"
OUT_DIR = DATA_ROOT / "teacher_features" / "fsc_full__hubert-large-ll60k__last_layer_mean"
SR = 16000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_split(ds, model, extractor):
    feats, labels, ids, metas = [], [], [], []
    for i in tqdm(range(len(ds)), desc="hubert-extract"):
        ex = ds[i]
        wav, _ = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32", always_2d=False)
        inputs = extractor(wav, sampling_rate=SR, return_tensors="pt")
        input_values = inputs["input_values"].to(DEVICE)
        out = model(input_values, output_hidden_states=True)
        last = out.hidden_states[-1][0]                               # [T', hidden]
        feat = last.mean(dim=0).float().cpu().to(torch.float16)       # [hidden]

        feats.append(feat)
        labels.append(int(ex["label_id"]))
        ids.append(str(ex.get("file", i)))
        metas.append({"num_samples": int(len(wav)), "num_frames": int(last.shape[0])})

    X = torch.stack(feats)                              # [N, hidden]
    return X, torch.tensor(labels, dtype=torch.long), ids, metas


def main():
    print(f"Device: {DEVICE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} (frozen, plain-SSL checkpoint -- no ASR/text fine-tuning)...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden_dim = model.config.hidden_size
    print(f"hidden_dim={hidden_dim}  num_layers={model.config.num_hidden_layers}")

    fsc, label2id = load_fsc()

    for split, fname in [("train", "train_features.pt"), ("validation", "val_features.pt")]:
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"exists, skip: {fname}")
            continue
        print(f"\n=== {split} ===")
        X, y, ids, metas = extract_split(fsc[split], model, extractor)
        print(f"{split}: {tuple(X.shape)}")
        torch.save({
            "labels": y, "sample_ids": ids, "metadata": metas,
            "feature_dim": hidden_dim,
            "features": {FEATURE_NAME: X},
        }, out_path)
        print(f"saved -> {out_path}")

    cfg = {
        "model": MODEL_NAME, "feature_name": FEATURE_NAME, "hidden_dim": hidden_dim,
        "pooling": "mean over time, final transformer layer (hidden_states[-1])",
        "splits_extracted": ["train", "validation"],
        "note": "test split intentionally not extracted (mirrors Qwen full_feature_extraction.py)",
    }
    with open(OUT_DIR / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
