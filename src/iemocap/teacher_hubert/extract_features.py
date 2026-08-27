"""
Pulls the HuBERT teacher KD targets. Frozen and LoRA-adapted, same script.

Mirrors iemocap/teacher/extract_features.py, and writes the same file schema,
so teacher_probe/probe_features.py and the student read these banks without
knowing which teacher produced them.

One forward pass per utterance gives:

    hubert_mean_l{12,18,24}  [1024]  frame mean at that encoder layer
    logits                   [C]     head output, the logit-KD target

Layer 18 is the one the student is distilled against. It is the same relative
depth as the Qwen feature, 27 of 36 against 18 of 24, and was fixed before any
probe was run; the other two are extracted so the probe can report what that
choice cost. There is no last_token counterpart, because there is no prompt and
no transcript to condition on - the privileged target is something the omni
teacher has and this one structurally cannot.

No sharding. HuBERT gets through a fold in a few minutes, so the resume
machinery the Qwen extractor needs would be dead weight.

    IEMOCAP_PROTOCOL=loso1 IEMOCAP_TEACHER=hubert \
        python src/iemocap/teacher_hubert/extract_features.py --split all \
            --adapter data/iemocap/teacher_qlora_hubert_loso1/hubert-large-ll60k_r64/adapter_ep3 \
            --head    data/iemocap/teacher_qlora_hubert_loso1/hubert-large-ll60k_r64/head_ep3.pt

    # frozen control, no adapter
    IEMOCAP_PROTOCOL=loso1 IEMOCAP_TEACHER=hubert \
        python src/iemocap/teacher_hubert/extract_features.py --split all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import PROTOCOL, TEACHER, IEMOCAP_FEATURES  # noqa: E402
from iemocap.teacher.data import (  # noqa: E402
    CLASSES, LABEL2ID, SPLITS, load_split, wav_path, load_audio, manifest_splits,
)
from iemocap.teacher_hubert.lora_finetune import MODEL_NAME, HubertClassifier, prep  # noqa: E402

LAYERS = (12, 18, 24)
STUDENT_LAYER = 18


def build_feat_tag(adapter_dir):
    base = "iemocap4__hubert-large-ll60k"
    return f"{base}__frozen" if adapter_dir is None else f"{base}-LORA__{adapter_dir.name}"


@torch.no_grad()
def extract_one(enc, head, fe, device, row):
    x = prep(fe, device, row)
    if x is None:
        raise ValueError("empty audio")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = enc(x, output_hidden_states=True, return_dict=True)
    feats = {f"hubert_mean_l{L}": out.hidden_states[L][0].mean(0).float().cpu().to(torch.float16)
             for L in LAYERS}
    if head is not None:
        # pooled the same way the fine-tune pooled it, or the logits would not be
        # the head that was trained
        z = out.last_hidden_state.mean(1)
        feats["logits"] = head(z.float()).float().cpu()[0].to(torch.float16)
    return feats, {"num_frames": int(out.last_hidden_state.shape[1])}


def process_split(enc, head, fe, device, df):
    fb, labels, ids, meta = {}, [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        feats, m = extract_one(enc, head, fe, device, row)
        for k, v in feats.items():
            fb.setdefault(k, []).append(v)
        labels.append(int(row["label"]))
        ids.append(row["turn_id"])
        m.update({"id": row["turn_id"], "label": row["emotion"], "label_id": int(row["label"]),
                  "speaker": row["speaker"], "is_impro": bool(row["is_impro"]),
                  "duration": float(row["duration"])})
        meta.append(m)
    return ({k: torch.stack(v) for k, v in fb.items()},
            torch.tensor(labels, dtype=torch.long), ids, meta)


def main():
    ap = argparse.ArgumentParser(description="Extract IEMOCAP HuBERT teacher features.")
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N samples (smoke test)")
    ap.add_argument("--adapter", default=None, help="adapter_epN/ -- omit for the frozen arm")
    ap.add_argument("--head", default=None, help="head_epN.pt, gives the logit-KD target")
    args = ap.parse_args()
    if TEACHER != "hubert":
        raise SystemExit("set IEMOCAP_TEACHER=hubert, otherwise this writes into the qwen tree")

    adapter_dir = Path(args.adapter) if args.adapter else None
    if adapter_dir is not None and args.head is None:
        raise SystemExit("--head is required with --adapter (it produces the logit-KD target)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    enc = HubertModel.from_pretrained(MODEL_NAME)
    enc.config.apply_spec_augment = False
    hidden = enc.config.hidden_size

    head = None
    if adapter_dir is not None:
        from peft import PeftModel
        enc = PeftModel.from_pretrained(enc, str(adapter_dir))
        head = HubertClassifier(enc, hidden, len(CLASSES)).head
        head.load_state_dict(torch.load(args.head, map_location="cpu"))
        head.to(device).eval()
    enc.to(device).eval()

    tag = build_feat_tag(adapter_dir)
    out_dir = IEMOCAP_FEATURES / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = manifest_splits() if args.split == "all" else (args.split,)
    print(f"{PROTOCOL} | {tag}\nsplits {splits} -> {out_dir}", flush=True)

    for split in splits:
        df = load_split(split, limit=args.limit)
        if df.empty:
            print(f"{split}: absent in this manifest, skipping")
            continue
        feats, labels, ids, meta = process_split(enc, head, fe, device, df)
        torch.save({"features": feats, "labels": labels, "sample_ids": ids, "metadata": meta,
                    "feature_dim": hidden}, out_dir / f"{split}_features.pt")
        print(f"  {split}: {len(ids)} x {sorted(feats)}", flush=True)

    with open(out_dir / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "protocol": PROTOCOL, "layers": list(LAYERS),
                   "student_layer": STUDENT_LAYER, "pooling": "frame mean",
                   "adapter": str(adapter_dir) if adapter_dir else None,
                   "head": args.head, "classes": CLASSES, "label2id": LABEL2ID,
                   "when": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
