"""
Bottleneck probe on the frozen HuBERT teacher.

Same B2 architecture and same protocol as fsc/teacher_probe/train_probe.py, so
this number sits directly next to the Qwen B2 row. No architecture sweep here,
only bottleneck=64, since that's the only one the student scripts use (the
feature-KD target has to match DSResNetSE.proj_dim).

Constants copied from the Qwen probe: 50 epochs, AdamW(1e-3, wd 1e-4), batch
256, dropout 0.1, CE, standardised with train mean/std. Acc and macro-F1
reported once at the end on FSC val. Like the Qwen probe, val is this stage's
final eval set and there is no separate test.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT, OUTPUTS_ROOT   # noqa: E402
from common.probe import Probe                      # noqa: E402

FEAT_TAG = "fsc_full__hubert-large-ll60k__last_layer_mean"
FEAT_DIR = DATA_ROOT / "teacher_features" / FEAT_TAG
FEATURE_NAME = "hubert_large_ll60k_last_layer_mean"

OUT_DATA = DATA_ROOT / "teacher_probe" / FEAT_TAG
CKPT_DIR = OUT_DATA / "checkpoints"
REP_DIR = OUT_DATA / "bottleneck_reps"
OUT_PLOT = OUTPUTS_ROOT / "fsc" / "hubert_baseline" / "teacher_probe"
for d in (CKPT_DIR, REP_DIR, OUT_PLOT):
    d.mkdir(parents=True, exist_ok=True)

# same hyperparameters as fsc/teacher_probe/train_probe.py
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
DROPOUT = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH_ID, HIDDEN, BOTTLENECK, ARCH_STR = "B2", [64], 64, "hidden->64->31"


def load_features():
    tr = torch.load(FEAT_DIR / "train_features.pt", weights_only=False)
    va = torch.load(FEAT_DIR / "val_features.pt", weights_only=False)
    Xtr = tr["features"][FEATURE_NAME].float()
    Xva = va["features"][FEATURE_NAME].float()
    ytr = tr["labels"].long()
    yva = va["labels"].long()

    # train stats only, same rule as the Qwen probe
    mean = Xtr.mean(dim=0, keepdim=True)
    std = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std
    return (Xtr_n, ytr, tr["sample_ids"]), (Xva_n, yva, va["sample_ids"]), (mean, std)


def main():
    print(f"Device: {DEVICE}")
    (Xtr, ytr, tr_ids), (Xva, yva, va_ids), (mean, std) = load_features()
    n_classes = int(max(ytr.max(), yva.max()).item()) + 1
    in_dim = Xtr.shape[1]
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  classes {n_classes}  in_dim {in_dim}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Probe(in_dim, HIDDEN, n_classes, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xva_d, yva_d = Xva.to(DEVICE), yva.to(DEVICE)
    n = Xtr_d.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_d[idx]), ytr_d[idx])
            loss.backward()
            opt.step()

    # eval once. no early stopping, no selection on val
    model.eval()
    with torch.no_grad():
        tr_pred = model(Xtr_d).argmax(1)
        va_pred = model(Xva_d).argmax(1)
    train_acc = (tr_pred == ytr_d).float().mean().item()
    eval_acc = (va_pred == yva_d).float().mean().item()
    eval_f1 = f1_score(yva.numpy(), va_pred.cpu().numpy(), average="macro")
    print(f"[{ARCH_ID}] {ARCH_STR}  eval_acc={eval_acc:.4f}  macroF1={eval_f1:.4f}  (train_acc={train_acc:.4f})")

    ckpt_path = CKPT_DIR / f"{ARCH_ID}_{ARCH_STR.replace('->', '-')}.pt"
    torch.save({
        "arch_id": ARCH_ID, "hidden_dims": HIDDEN, "n_classes": n_classes,
        "arch_str": ARCH_STR, "dropout": DROPOUT,
        "state_dict": model.state_dict(),
        "standardizer": {"mean": mean, "std": std},
        "feature_name": FEATURE_NAME, "in_dim": in_dim,
        "eval_acc": eval_acc, "eval_macro_f1": eval_f1, "train_acc": train_acc,
    }, ckpt_path)
    print(f"Checkpoint -> {ckpt_path}")

    # export the bottleneck, this is what the student distills from
    model.eval()
    with torch.no_grad():
        _, btr = model(Xtr_d, return_bottleneck=True)
        _, bva = model(Xva_d, return_bottleneck=True)
    rep_path = REP_DIR / f"{ARCH_ID}_bottleneck{BOTTLENECK}.pt"
    torch.save({
        "arch_id": ARCH_ID, "bottleneck_dim": BOTTLENECK,
        "train": {"emb": btr.cpu().to(torch.float16), "labels": ytr, "sample_ids": tr_ids},
        "val": {"emb": bva.cpu().to(torch.float16), "labels": yva, "sample_ids": va_ids},
        "note": "bottleneck activation (post LayerNorm-GELU); teacher rep for KD (HuBERT audio-only baseline)",
    }, rep_path)
    print(f"Bottleneck reps -> {rep_path}")

    csv_path = OUT_PLOT / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "arch", "bottleneck", "eval_acc", "eval_macro_f1", "train_acc"])
        w.writeheader()
        w.writerow({"id": ARCH_ID, "arch": ARCH_STR, "bottleneck": BOTTLENECK,
                    "eval_acc": round(eval_acc, 4), "eval_macro_f1": round(eval_f1, 4),
                    "train_acc": round(train_acc, 4)})
    print(f"Results CSV -> {csv_path}")

    md = ["| ID | Architecture | Bottleneck | Eval Acc | Eval Macro F1 | Train Acc |",
          "| -- | ------------ | ---------- | -------- | ------------- | --------- |",
          f"| {ARCH_ID} | {ARCH_STR} | {BOTTLENECK} | {eval_acc:.4f} | {eval_f1:.4f} | {train_acc:.4f} |"]
    (OUT_PLOT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Results MD -> {OUT_PLOT / 'results.md'}")
    print("Done.")


if __name__ == "__main__":
    main()
