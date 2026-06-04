"""
Student KD ablation: same DSResNet-SE, four training losses.

The key question is NOT architecture comparison but whether distilling the
Qwen2.5-Omni teacher (64-dim bottleneck + logits) improves the same compact
audio-only student:

    CE-only   :  L_ce
    Logit-KD  :  L_ce + lam_logit * L_logit
    Feature-KD:  L_ce + lam_feature * L_feature
    Full-KD   :  L_ce + lam_logit * L_logit + lam_feature * L_feature

Teacher signals are produced from the chosen B2 probe (2048->64->31):
  - teacher_z_64   = B2 bottleneck activation
  - teacher_logits = B2 classifier logits
computed on the cached teacher features (FSC order), aligned to the student
log-mel cache by file id.

Inputs:
  data/student/logmel_cache/{train,val}_logmel.pt          (precompute_logmel.py)
  data/teacher_features/<feat>/{train,val}_features.pt      (full_feature_extraction.py)
  data/teacher_probe/<feat>/checkpoints/B2_*.pt            (train_probe.py)

Outputs:
  data/student/checkpoints/<exp>_best.pt
  data/student/results.csv
  outputs/student/student_kd_ablation.png
  outputs/student/results.md
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

PROJECT = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
sys.path.insert(0, str(PROJECT / "src" / "student"))
sys.path.insert(0, str(PROJECT / "src" / "teacher_probe"))
from student_model import DSResNetSE, model_summary   # noqa: E402
from train_probe import Probe                          # noqa: E402  (teacher probe class)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA     = PROJECT / "data"
LOGMEL   = DATA / "student" / "logmel_cache"
FEAT_TAG = "fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34"
FEAT_DIR = DATA / "teacher_features" / FEAT_TAG
PROBE_CKPT_DIR = DATA / "teacher_probe" / FEAT_TAG / "checkpoints"
CKPT_DIR = DATA / "student" / "checkpoints"
OUT_PLOT = PROJECT / "outputs" / "student"
for d in (CKPT_DIR, OUT_PLOT):
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────────
# Matches the strong CE-only baseline (baseline.py 'arch_reg', val ~0.884):
# SpecAugment + label smoothing are part of the shared training setup, applied
# identically to all four ablations so the only variable is the KD loss.
EPOCHS      = 70
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 256
DROPOUT     = 0.2
LABEL_SMOOTH = 0.1
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# KD defaults (README)
T           = 2.0
LAM_LOGIT   = 0.5
LAM_FEATURE = 1.0

EXPERIMENTS = ["ce_only", "logit_kd", "feature_kd", "full_kd"]
USE_LOGIT   = {"ce_only": False, "logit_kd": True,  "feature_kd": False, "full_kd": True}
USE_FEATURE = {"ce_only": False, "logit_kd": False, "feature_kd": True,  "full_kd": True}


# ── Teacher signals ──────────────────────────────────────────────────────────────
def build_teacher_signals():
    """Return dict split -> (z_64 [N,64], logits [N,31], sample_ids) from the B2 probe."""
    ckpt_path = next(PROBE_CKPT_DIR.glob("B2_*.pt"))
    ckpt = torch.load(ckpt_path, weights_only=False)
    mean, std = ckpt["standardizer"]["mean"], ckpt["standardizer"]["std"]
    feat_name = ckpt["feature_name"]

    probe = Probe(2048, ckpt["hidden_dims"], ckpt["n_classes"], dropout=ckpt["dropout"])
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval().to(DEVICE)

    out = {}
    for split, fname in [("train", "train_features.pt"), ("val", "val_features.pt")]:
        d = torch.load(FEAT_DIR / fname, weights_only=False)
        X = ((d["features"][feat_name].float() - mean) / std).to(DEVICE)
        with torch.no_grad():
            logits, z = probe(X, return_bottleneck=True)
        out[split] = (z.cpu(), logits.cpu(), d["sample_ids"])
    return out, ckpt_path.name


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    tr = torch.load(LOGMEL / "train_logmel.pt", weights_only=False)
    va = torch.load(LOGMEL / "val_logmel.pt",   weights_only=False)
    mean = tr["mean"].view(1, 1, 1, -1)
    std  = tr["std"].view(1, 1, 1, -1)

    Xtr = (tr["logmel"].float() - mean) / std
    Xva = (va["logmel"].float() - mean) / std
    ytr, yva = tr["labels"].long(), va["labels"].long()

    teacher, probe_name = build_teacher_signals()
    ztr, ltr, idtr = teacher["train"]
    zva, lva, idva = teacher["val"]

    # Critical: teacher signals and student inputs must be the same samples, same order.
    assert tr["sample_ids"] == idtr, "TRAIN sample_id mismatch (student vs teacher)"
    assert va["sample_ids"] == idva, "VAL sample_id mismatch (student vs teacher)"

    print(f"Teacher probe   : {probe_name}")
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")
    return (Xtr, ytr, ztr, ltr), (Xva, yva, zva, lva)


# ── Losses ──────────────────────────────────────────────────────────────────────
def kd_logit_loss(student_logits, teacher_logits, t=T):
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def kd_feature_loss(student_z, teacher_z):
    return 1.0 - F.cosine_similarity(student_z, teacher_z, dim=1).mean()


def spec_augment(x, n_freq=2, n_time=2, f_max=12, t_max=40):
    """Per-batch time/freq masking (training only). x: [B,1,T,F] normalized log-mel."""
    B, _, T, F_ = x.shape
    x = x.clone()
    for _ in range(n_freq):
        f = int(torch.randint(0, f_max + 1, (1,)))
        if f > 0:
            f0 = int(torch.randint(0, max(1, F_ - f), (1,)))
            x[:, :, :, f0:f0 + f] = 0.0
    for _ in range(n_time):
        t = int(torch.randint(0, t_max + 1, (1,)))
        if t > 0:
            t0 = int(torch.randint(0, max(1, T - t), (1,)))
            x[:, :, t0:t0 + t, :] = 0.0
    return x


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    preds = []
    for i in range(0, X.shape[0], BATCH_SIZE):
        _, logits = model(X[i:i + BATCH_SIZE].to(DEVICE))
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    yt = y.numpy()
    return {
        "acc": accuracy_score(yt, preds),
        "macro_f1": f1_score(yt, preds, average="macro"),
        "weighted_f1": f1_score(yt, preds, average="weighted"),
    }


# ── Train one experiment ──────────────────────────────────────────────────────────
def train_experiment(exp, train_data, val_data):
    Xtr, ytr, ztr, ltr = train_data
    Xva, yva, _, _ = val_data
    use_logit, use_feature = USE_LOGIT[exp], USE_FEATURE[exp]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = DSResNetSE(dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    n = Xtr.shape[0]
    best_f1, best_state, best_metrics, best_epoch = -1.0, None, None, -1
    history = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = spec_augment(Xtr[idx].to(DEVICE))
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            z_s, logits_s = model(xb)
            loss = ce(logits_s, yb)
            if use_logit:
                loss = loss + LAM_LOGIT * kd_logit_loss(logits_s, ltr[idx].to(DEVICE))
            if use_feature:
                loss = loss + LAM_FEATURE * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        sched.step()

        m = evaluate(model, Xva, yva)
        history.append({"epoch": epoch, "train_loss": running / n, **m})
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_metrics = m
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  loss={running/n:.4f}  "
                  f"val_acc={m['acc']:.4f}  macroF1={m['macro_f1']:.4f}")

    # Save best-by-val-macroF1 checkpoint (README).
    torch.save({"exp": exp, "state_dict": best_state, "best_epoch": best_epoch,
                "metrics": best_metrics, "history": history},
               CKPT_DIR / f"{exp}_best.pt")
    return best_metrics, best_epoch, history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    sz = model_summary(DSResNetSE())
    print(f"Student params: {sz['params']:,}  | FP32 {sz['fp32_mb']:.2f} MB  | INT8(est) {sz['int8_mb']:.2f} MB\n")

    train_data, val_data = load_data()

    results = []
    for exp in EXPERIMENTS:
        print(f"\n=== {exp} ===")
        metrics, best_epoch, _ = train_experiment(exp, train_data, val_data)
        print(f"  BEST @epoch {best_epoch+1}: acc={metrics['acc']:.4f}  "
              f"macroF1={metrics['macro_f1']:.4f}  weightedF1={metrics['weighted_f1']:.4f}")
        results.append({"exp": exp, **metrics,
                        "params": sz["params"], "fp32_mb": round(sz["fp32_mb"], 2),
                        "int8_mb": round(sz["int8_mb"], 2)})

    # ── results CSV + MD ─────────────────────────────────────────────────────────
    csv_path = DATA / "student" / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "acc", "macro_f1", "weighted_f1",
                                          "params", "fp32_mb", "int8_mb"])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")

    md = ["| Experiment | Val Acc | Macro F1 | Weighted F1 | Params | FP32 MB | INT8 MB |",
          "| ---------- | ------- | -------- | ----------- | ------ | ------- | ------- |"]
    for r in results:
        md.append(f"| {r['exp']} | {r['acc']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} "
                  f"| {r['params']:,} | {r['fp32_mb']} | {r['int8_mb']} |")
    (OUT_PLOT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))

    make_plot(results)


def make_plot(results):
    exps = [r["exp"] for r in results]
    metrics = [("acc", "Val Accuracy", "#1f77b4"),
               ("macro_f1", "Macro F1", "#ff7f0e"),
               ("weighted_f1", "Weighted F1", "#2ca02c")]

    vals_all = [r[m] for r in results for m, _, _ in metrics]
    lo, hi = min(vals_all) - 0.01, max(vals_all) + 0.008

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(exps))
    w = 0.26
    for j, (key, label, color) in enumerate(metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.0009,
                    f"{b.get_height():.4f}", ha="center", va="top",
                    fontsize=8, rotation=90, color="white", fontweight="bold")

    # CE-only reference line (the question: does KD beat it?)
    ce_acc = results[0]["acc"]
    ax.axhline(ce_acc, color="gray", linestyle=":", linewidth=1.3,
               label=f"CE-only acc ({ce_acc:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=10)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Score (FSC validation, 31 classes)")
    ax.set_title("Student KD Ablation — DSResNet-SE (379K params, FP32 1.45 MB) + SpecAugment + label-smooth\n"
                 "audio-only student  ·  teacher = Qwen2.5-Omni-3B B2 bottleneck(64) + logits  "
                 "·  (y-axis zoomed)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.35)

    plot_path = OUT_PLOT / "student_kd_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot        -> {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
