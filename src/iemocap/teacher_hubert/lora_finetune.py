"""
LoRA fine-tune of HuBERT-large for 4-class IEMOCAP emotion.

The audio-only teacher control. Qwen2.5-Omni reads the transcript during its
fine-tune, this one cannot read text at all, so distilling both into the same
student separates "a strong teacher representation helps" from "an omni-modal
teacher helps". FSC ran the same control with a frozen HuBERT (chapter 4), but
here both teachers are adapted, so the comparison is at a matched adaptation
budget rather than frozen against fine-tuned.

facebook/hubert-large-ll60k is the plain SSL checkpoint, masked prediction over
unlabelled Libri-Light. The -ls960-ft variant would put CTC transcripts back in
and defeat the point.

Every choice that could have gone either way is copied from
iemocap/teacher/lora_finetune.py: 3 epochs, lr 2e-4 adapter and 1e-3 head,
batch 1 with 16-step accumulation, cosine schedule with warmup, the same head
shape at 1024 -> 256 -> C, and the whole utterance rather than the student 8 s
crop. Rank 64 is the rank the Qwen audio tower got. What differs is forced by
the model: no prompt, so no instruction and no readout cue, and pooling is a
frame mean instead of a last-token readout.

Module names differ too, and that matters more than it looks. The Qwen target
list uses Qwen and Whisper naming and would match only q/k/v/out_proj here,
leaving every mlp unadapted, which is the MIntRec adapter bug in a new place.
So the targets are written for HuBERT and the trainable count is printed before
training starts.

No epoch is selected. LOSO has no val split and the Qwen teacher used its
third-epoch adapter everywhere, so this one does too. The held-out session is
scored after every epoch and logged, for monitoring only.

    IEMOCAP_PROTOCOL=loso1 IEMOCAP_TEACHER=hubert \
        python src/iemocap/teacher_hubert/lora_finetune.py --limit 40 --epochs 1
    IEMOCAP_PROTOCOL=loso1 IEMOCAP_TEACHER=hubert \
        python src/iemocap/teacher_hubert/lora_finetune.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import PROTOCOL, SUF, TEACHER, IEMOCAP_QLORA, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.teacher.data import (  # noqa: E402
    CLASSES, LABEL2ID, load_split, wav_path, load_audio, manifest_splits,
)

MODEL_NAME = "facebook/hubert-large-ll60k"
# attention plus both feed-forward projections. qwen calls the mlp pair fc1 and
# fc2, hubert calls them intermediate_dense and output_dense
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj",
                "intermediate_dense", "output_dense"]


class HubertClassifier(nn.Module):
    """LoRA encoder + frame mean -> MLP head. Same head shape as OmniClassifier."""

    def __init__(self, enc, hidden, n_classes, head_hidden=256, dropout=0.2):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.LayerNorm(head_hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, n_classes),
        )

    def forward(self, x):
        h = self.enc(x).last_hidden_state          # [1, T, H], batch 1 so no mask
        return self.head(h.mean(1))


def describe_trainable(model):
    """peft matches by module name, so count what actually got an adapter before
    spending an hour training something half-covered."""
    by, trainable = {}, 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        trainable += p.numel()
        m = re.search(r"\.([a-z_]+)\.lora_[AB]", n)
        k = m.group(1) if m else "other"
        by[k] = by.get(k, 0) + p.numel()
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f} M / {total/1e6:.1f} M  "
          + ", ".join(f"{k}={v/1e6:.2f}M" for k, v in sorted(by.items(), key=lambda x: -x[1])))
    missing = [t for t in LORA_TARGETS if t not in by]
    if missing:
        print(f"  WARNING: no adapter matched {missing}")
    return trainable, total


def prep(fe, device, row):
    wav = load_audio(wav_path(row))
    if wav.size == 0:
        return None
    x = fe(wav, sampling_rate=16000, return_tensors="pt")["input_values"]
    return x.to(device)


@torch.no_grad()
def evaluate(clf, fe, device, df, desc="eval"):
    clf.eval()
    ys, ps = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        x = prep(fe, device, row)
        if x is None:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = clf(x)
        ps.append(int(logits.argmax(-1)[0]))
        ys.append(int(row["label"]))
    ys, ps = np.array(ys), np.array(ps)
    return {
        "n": int(len(ys)),
        "wa": float((ps == ys).mean()),                       # weighted acc = plain acc
        "ua": float(recall_score(ys, ps, average="macro")),   # unweighted = macro recall
        "macro_f1": float(f1_score(ys, ps, average="macro")),
        "per_class_recall": recall_score(ys, ps, average=None,
                                         labels=list(range(len(CLASSES)))).round(4).tolist(),
        "confusion": confusion_matrix(ys, ps, labels=list(range(len(CLASSES)))).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune HuBERT-large on IEMOCAP.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=16,
                    help="grad-accumulation steps (physical batch=1)")
    ap.add_argument("--lora-r", type=int, default=64, help="the rank the qwen audio tower got")
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N train samples (smoke)")
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if TEACHER != "hubert":
        raise SystemExit("set IEMOCAP_TEACHER=hubert, otherwise this writes into the qwen tree")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"
    train_df = load_split("train", limit=args.limit)
    # no val under LOSO. the held-out session gets scored every epoch for
    # monitoring, it never picks an epoch
    watch = "val" if "val" in manifest_splits() else "test"
    watch_df = load_split(watch, limit=args.eval_limit)
    print(f"{PROTOCOL} | train {len(train_df)} | watching {watch} {len(watch_df)}")

    print(f"Loading {MODEL_NAME} ...", flush=True)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    enc = HubertModel.from_pretrained(MODEL_NAME)
    enc.config.apply_spec_augment = False        # augmentation belongs to the student
    hidden = enc.config.hidden_size
    enc = get_peft_model(enc, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", target_modules=LORA_TARGETS))
    n_train_p, n_total_p = describe_trainable(enc)

    clf = HubertClassifier(enc, hidden, len(CLASSES), head_hidden=args.head_hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([
        {"params": [p for p in enc.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": clf.head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    steps = max((len(train_df) * args.epochs) // args.accum, 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

    tag = f"hubert-large-ll60k_r{args.lora_r}"
    out_dir = IEMOCAP_QLORA / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = IEMOCAP_OUTPUTS / "teacher_hubert"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"lora_{tag}{SUF}.csv"
    print(f"OUT: {out_dir}\nLOG: {log_csv}", flush=True)

    rows = []
    for ep in range(args.epochs):
        clf.train()
        opt.zero_grad()
        order = np.random.permutation(len(train_df))
        running, seen = 0.0, 0
        pbar = tqdm(order, desc=f"epoch {ep+1}/{args.epochs}")
        for step, ridx in enumerate(pbar):
            row = train_df.iloc[int(ridx)]
            try:
                x = prep(fe, device, row)
                if x is None:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = clf(x)
                y = torch.tensor([int(row["label"])], device=device)
                loss = loss_fn(logits.float(), y) / args.accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                print("  OOM skip", row["turn_id"], f"dur={row['duration']:.1f}s")
                opt.zero_grad()
                torch.cuda.empty_cache()
                continue
            running += loss.item() * args.accum
            seen += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in enc.parameters() if p.requires_grad]
                    + list(clf.head.parameters()), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                pbar.set_postfix(loss=f"{running/max(seen,1):.3f}",
                                 gb=f"{torch.cuda.max_memory_allocated()/1e9:.1f}")

        m = evaluate(clf, fe, device, watch_df, desc=watch)
        print(f"[epoch {ep+1}] train_loss={running/max(seen,1):.4f}  "
              f"{watch} WA={m['wa']:.4f}  UA={m['ua']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
        print(f"  per-class recall {dict(zip(CLASSES, m['per_class_recall']))}", flush=True)
        enc.save_pretrained(out_dir / f"adapter_ep{ep+1}")
        torch.save(clf.head.state_dict(), out_dir / f"head_ep{ep+1}.pt")
        rows.append({"protocol": PROTOCOL, "epoch": ep + 1,
                     "train_loss": round(running / max(seen, 1), 4),
                     "watch_split": watch, "n": m["n"], "wa": m["wa"], "ua": m["ua"],
                     "macro_f1": m["macro_f1"],
                     **{f"recall_{c}": v for c, v in zip(CLASSES, m["per_class_recall"])},
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        pd.DataFrame(rows).to_csv(log_csv, index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "model_name": MODEL_NAME, "hidden": hidden,
                   "protocol": PROTOCOL, "n_classes": len(CLASSES), "classes": CLASSES,
                   "label2id": LABEL2ID, "lora_targets": LORA_TARGETS,
                   "trainable_params": n_train_p, "total_params": n_total_p,
                   "pool": "frame_mean", "watch_split": watch,
                   "selection_metric": "none (last epoch, as in the qwen teacher)",
                   "epochs_log": rows}, f, indent=2, ensure_ascii=False)
    print(f"Saved adapters + heads + config -> {out_dir}")


if __name__ == "__main__":
    main()
