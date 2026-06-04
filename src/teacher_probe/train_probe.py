"""
Teacher-probe bottleneck experiments on the full-FSC extracted feature.

Trains 7 probe heads on top of the frozen teacher feature
    prompt_first_audio_mean_L24-27-30-34   ([N, 2048], fp16)
to answer: how small a bottleneck can we use before the teacher's intent
information degrades?  The chosen bottleneck dim becomes the student's KD target.

Protocol (per README — strict, no leakage):
  * FSC validation is the FINAL eval set -> NO early stopping / model selection on it.
  * Fixed schedule: 50 epochs, AdamW(lr=1e-3, wd=1e-4), batch 256, dropout 0.1, CE loss.
  * Standardize with TRAIN mean/std (applied to both train and eval).
  * Report eval accuracy + macro F1 once, after training. Save final checkpoint.

Architectures:
  A1  2048 -> 31              linear baseline
  A2  2048 -> 1024 -> 31      nonlinear upper bound
  A3  2048 -> 2048 -> 31      full-dim MLP upper bound
  B1  2048 -> 32  -> 31       compact bottleneck
  B2  2048 -> 64  -> 31       likely sweet spot
  B3  2048 -> 128 -> 31       stable compact teacher
  B4  2048 -> 256 -> 31       higher-capacity bottleneck

MLP block = Linear -> LayerNorm -> GELU -> Dropout(0.1); final Linear(d, 31).

Outputs:
  data/teacher_probe/<feat>/checkpoints/<ID>_<arch>.pt    (state_dict + standardizer + cfg)
  data/teacher_probe/<feat>/bottleneck_reps/<ID>_bottleneck<d>.pt   (B* only; train+val embeddings)
  data/teacher_probe/<feat>/results.csv
  outputs/teacher_probe/probe_comparison.png
  outputs/teacher_probe/results.md
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT  = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
FEAT_TAG = "fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34"
FEAT_DIR = PROJECT / "data" / "teacher_features" / FEAT_TAG
FEATURE_NAME = "prompt_first_audio_mean_L24-27-30-34"

OUT_DATA = PROJECT / "data" / "teacher_probe" / FEAT_TAG
CKPT_DIR = OUT_DATA / "checkpoints"
REP_DIR  = OUT_DATA / "bottleneck_reps"
OUT_PLOT = PROJECT / "outputs" / "teacher_probe"
for d in (CKPT_DIR, REP_DIR, OUT_PLOT):
    d.mkdir(parents=True, exist_ok=True)

# ── Fixed hyperparameters (README) ──────────────────────────────────────────────
EPOCHS      = 50
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 256
DROPOUT     = 0.1
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# id, hidden dims, bottleneck dim to export (None = don't export), arch str, description
ARCHS = [
    ("A1", [],        None, "2048->31",          "linear baseline"),
    ("A2", [1024],    None, "2048->1024->31",    "nonlinear upper bound"),
    ("A3", [2048],    None, "2048->2048->31",    "full-dim MLP upper bound"),
    ("B1", [32],      32,   "2048->32->31",      "compact bottleneck"),
    ("B2", [64],      64,   "2048->64->31",      "likely sweet spot"),
    ("B3", [128],     128,  "2048->128->31",     "stable compact teacher"),
    ("B4", [256],     256,  "2048->256->31",     "higher-capacity bottleneck"),
    # Deeper compression (2048 -> 512 -> d -> 31); bottleneck = the final hidden dim d.
    ("C1", [512, 32], 32,   "2048->512->32->31", "deep compression"),
    ("C2", [512, 64], 64,   "2048->512->64->31", "deep compression"),
    ("C3", [512, 128],128,  "2048->512->128->31","deep compression"),
]


# ── Model ───────────────────────────────────────────────────────────────────────
class Probe(nn.Module):
    """MLP probe; bottleneck = activation of the last hidden block (feeds the head)."""

    def __init__(self, in_dim, hidden_dims, n_classes, dropout=DROPOUT):
        super().__init__()
        self.blocks = nn.ModuleList()
        d = in_dim
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(
                nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)
            ))
            d = h
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, return_bottleneck=False):
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        if return_bottleneck:
            return logits, x
        return logits


# ── Data ──────────────────────────────────────────────────────────────────────
def load_features():
    tr = torch.load(FEAT_DIR / "train_features.pt", weights_only=False)
    va = torch.load(FEAT_DIR / "val_features.pt",   weights_only=False)
    Xtr = tr["features"][FEATURE_NAME].float()
    Xva = va["features"][FEATURE_NAME].float()
    ytr = tr["labels"].long()
    yva = va["labels"].long()

    # Standardize with TRAIN statistics only.
    mean = Xtr.mean(dim=0, keepdim=True)
    std  = Xtr.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std

    return (Xtr_n, ytr, tr["sample_ids"]), (Xva_n, yva, va["sample_ids"]), (mean, std)


# ── Train / eval one architecture ────────────────────────────────────────────────
def train_one(arch_id, hidden, Xtr, ytr, Xva, yva, n_classes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = Probe(Xtr.shape[1], hidden, n_classes).to(DEVICE)
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

    # Final eval (once).
    model.eval()
    with torch.no_grad():
        tr_pred = model(Xtr_d).argmax(1)
        va_pred = model(Xva_d).argmax(1)
    train_acc = (tr_pred == ytr_d).float().mean().item()
    eval_acc  = (va_pred == yva_d).float().mean().item()
    eval_f1   = f1_score(yva.numpy(), va_pred.cpu().numpy(), average="macro")

    return model, train_acc, eval_acc, eval_f1


# ── Run all ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    (Xtr, ytr, tr_ids), (Xva, yva, va_ids), (mean, std) = load_features()
    n_classes = int(max(ytr.max(), yva.max()).item()) + 1
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  classes {n_classes}\n")

    results = []
    for arch_id, hidden, bottleneck, arch_str, desc in ARCHS:
        print(f"[{arch_id}] {arch_str:<18} ({desc}) ... ", end="", flush=True)
        model, tr_acc, ev_acc, ev_f1 = train_one(
            arch_id, hidden, Xtr, ytr, Xva, yva, n_classes
        )
        print(f"eval_acc={ev_acc:.4f}  macroF1={ev_f1:.4f}  (train_acc={tr_acc:.4f})")

        # Save checkpoint (with standardizer so the probe is reusable on raw features).
        ckpt_path = CKPT_DIR / f"{arch_id}_{arch_str.replace('->', '-')}.pt"
        torch.save({
            "arch_id": arch_id, "hidden_dims": hidden, "n_classes": n_classes,
            "arch_str": arch_str, "dropout": DROPOUT,
            "state_dict": model.state_dict(),
            "standardizer": {"mean": mean, "std": std},
            "feature_name": FEATURE_NAME,
            "eval_acc": ev_acc, "eval_macro_f1": ev_f1, "train_acc": tr_acc,
        }, ckpt_path)

        # Export bottleneck representations for B* (candidate student KD targets).
        if bottleneck is not None:
            model.eval()
            with torch.no_grad():
                _, btr = model(Xtr.to(DEVICE), return_bottleneck=True)
                _, bva = model(Xva.to(DEVICE), return_bottleneck=True)
            rep_path = REP_DIR / f"{arch_id}_bottleneck{bottleneck}.pt"
            torch.save({
                "arch_id": arch_id, "bottleneck_dim": bottleneck,
                "train": {"emb": btr.cpu().to(torch.float16), "labels": ytr, "sample_ids": tr_ids},
                "val":   {"emb": bva.cpu().to(torch.float16), "labels": yva, "sample_ids": va_ids},
                "note": "bottleneck activation (post LayerNorm-GELU); candidate teacher rep for KD",
            }, rep_path)

        results.append({
            "id": arch_id, "arch": arch_str, "bottleneck": bottleneck if bottleneck else "-",
            "eval_acc": ev_acc, "eval_macro_f1": ev_f1, "train_acc": tr_acc,
        })

    # ── Save results CSV ─────────────────────────────────────────────────────────
    csv_path = OUT_DATA / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "arch", "bottleneck",
                                          "eval_acc", "eval_macro_f1", "train_acc"])
        w.writeheader()
        w.writerows(results)
    print(f"\nResults CSV  -> {csv_path}")

    # ── Save results markdown table ──────────────────────────────────────────────
    md = ["| ID | Architecture | Bottleneck | Eval Acc | Eval Macro F1 | Train Acc |",
          "| -- | ------------ | ---------- | -------- | ------------- | --------- |"]
    for r in results:
        md.append(f"| {r['id']} | {r['arch']} | {r['bottleneck']} | "
                  f"{r['eval_acc']:.4f} | {r['eval_macro_f1']:.4f} | {r['train_acc']:.4f} |")
    md_path = OUT_PLOT / "results.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Results MD   -> {md_path}")

    print("\n" + "\n".join(md))

    # ── Plot ─────────────────────────────────────────────────────────────────────
    make_plot(results)


def make_plot(results):
    ids   = [r["id"] for r in results]
    accs  = [r["eval_acc"] for r in results]
    f1s   = [r["eval_macro_f1"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.55, 1]})

    # Zoomed y-range so the (very close) scores are actually distinguishable.
    vmin = min(min(accs), min(f1s))
    vmax = max(max(accs), max(f1s))
    lo, hi = vmin - 0.006, vmax + 0.006

    # Left: grouped bars (eval acc + macro F1) for all architectures.
    x = np.arange(len(ids))
    w = 0.40
    bars1 = ax1.bar(x - w/2, accs, w, label="Eval Accuracy", color="#1f77b4")
    bars2 = ax1.bar(x + w/2, f1s, w, label="Eval Macro F1", color="#ff7f0e")
    # Vertical value labels just under each bar top -> no horizontal overlap.
    for bars in (bars1, bars2):
        for b in bars:
            ax1.text(b.get_x() + b.get_width()/2, b.get_height() - 0.0008,
                     f"{b.get_height():.4f}", ha="center", va="top",
                     fontsize=7.5, rotation=90, color="white", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r['id']}\n{r['arch']}" for r in results],
                        fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Score (31 intent classes)")
    ax1.set_ylim(lo, hi)
    ax1.set_title("Teacher Probe Comparison — Eval Accuracy & Macro F1   (y-axis zoomed)\n"
                  "Qwen2.5-Omni-3B  ·  pf_audio_mean L[24,27,30,34]  ·  FSC full (train 23132 / eval 3118)",
                  fontsize=9.5)
    ax1.axhline(accs[0], color="gray", linestyle=":", linewidth=1.2,
                label=f"A1 linear ({accs[0]:.4f})")
    # Highlight the best eval-acc architecture.
    best_i = int(np.argmax(accs))
    ax1.annotate("best", (x[best_i] - w/2, accs[best_i]), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=8, fontweight="bold", color="#1f77b4")
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.grid(axis="y", alpha=0.35)

    # Right: bottleneck dim vs eval acc (the compression tradeoff) + upper-bound refs.
    # Two curves: B* (single-layer compression) vs C* (deep 512-then-d compression).
    b_dims = [r["bottleneck"] for r in results if r["id"].startswith("B")]
    b_accs = [r["eval_acc"]   for r in results if r["id"].startswith("B")]
    c_dims = [r["bottleneck"] for r in results if r["id"].startswith("C")]
    c_accs = [r["eval_acc"]   for r in results if r["id"].startswith("C")]

    ax2.plot(b_dims, b_accs, "o-", color="#2ca02c", linewidth=2, markersize=7,
             label="B* single-layer (2048->d)")
    for d_, a_ in zip(b_dims, b_accs):
        ax2.annotate(f"{a_:.3f}", (d_, a_), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#2ca02c")
    if c_dims:
        ax2.plot(c_dims, c_accs, "s--", color="#8c564b", linewidth=2, markersize=7,
                 label="C* deep (2048->512->d)")
        for d_, a_ in zip(c_dims, c_accs):
            ax2.annotate(f"{a_:.3f}", (d_, a_), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=8, color="#8c564b")

    # reference lines
    a_map = {r["id"]: r["eval_acc"] for r in results}
    ax2.axhline(a_map["A1"], color="gray",   linestyle=":",  linewidth=1.2, label=f"A1 linear ({a_map['A1']:.3f})")
    ax2.axhline(a_map["A2"], color="#d62728", linestyle="--", linewidth=1.2, label=f"A2 1024 upper ({a_map['A2']:.3f})")
    ax2.axhline(a_map["A3"], color="#9467bd", linestyle="--", linewidth=1.2, label=f"A3 2048 upper ({a_map['A3']:.3f})")
    all_dims = sorted(set(b_dims) | set(c_dims))
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(all_dims)
    ax2.set_xticklabels([str(d_) for d_ in all_dims])
    # Zoom right axis to the bottleneck/upper-bound region so the curves separate.
    rt_vals = b_accs + c_accs + [a_map["A1"], a_map["A2"], a_map["A3"]]
    ax2.set_ylim(min(rt_vals) - 0.006, max(rt_vals) + 0.006)
    ax2.set_xlabel("Bottleneck dimension")
    ax2.set_ylabel("Eval Accuracy   (y-axis zoomed)")
    ax2.set_title("Compression Tradeoff: bottleneck dim vs eval accuracy", fontsize=9.5)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.3)

    plot_path = OUT_PLOT / "probe_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot         -> {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
