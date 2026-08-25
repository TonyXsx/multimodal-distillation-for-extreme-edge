"""
Frozen HuBERT-large as an audio-only teacher. This is the control for the Qwen
one, so the KD comparison can separate "any strong frozen teacher helps" from
"the prompted multimodal teacher helps".

Model is facebook/hubert-large-ll60k, the plain SSL checkpoint (masked
prediction of kmeans cluster ids over 60k h of unlabelled Libri-Light). No text
or ASR fine-tuning anywhere in its history. The -ls960-ft variant would be
wrong here, CTC fine-tuning on transcripts puts text supervision back in and
kills the point of an audio-only control.

Feature is the final layer, mean-pooled over time. No layer or prompt-order
ablation this time, the Qwen side already answered that and this baseline is
deliberately scoped not to repeat it. Standard frozen-SSL pooling, same recipe
SUPERB uses.

Train and val only. Test stays teacher-free for the same reason as in
full_feature_extraction.py.

Sample order and ids come from precompute_logmel.load_fsc() by import rather
than being redone, so the ids line up with the existing logmel cache.

Same output schema as the Qwen banks.
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
        feat = last.mean(dim=0).float().cpu().to(torch.float16)

        feats.append(feat)
        labels.append(int(ex["label_id"]))
        ids.append(str(ex.get("file", i)))
        metas.append({"num_samples": int(len(wav)), "num_frames": int(last.shape[0])})

    X = torch.stack(feats)
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
