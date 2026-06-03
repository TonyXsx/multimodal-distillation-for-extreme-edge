"""
Feature combination ablation: mean vs concat over layers [24, 27, 30, 34].

Tests whether combining multiple layers improves over the best single layer,
within each of three feature families (no cross-family mixing):

  1. prompt_first · audio_mean   (main KD target candidate)
  2. audio_first  · audio_mean   (audio-side control)
  3. audio_first  · last_text    (task-aware upper bound)

Two combination methods per family:
  mean   : average 4 layer vectors  -> [N, 2048]   (same dim as single layer)
  concat : concatenate 4 vectors    -> [N, 8192]   (4x dim)

Same logistic regression probe as linear_probe.py for fair comparison.

Outputs:
  outputs/feature_ablation/feature_combinations/
    feature_combinations.png
    results.csv
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT  = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
FEAT_DIR = PROJECT / "data" / "teacher_features" / "fsc_small_ablation__qwen2.5-omni-3b-4bit"
OUT_DIR  = PROJECT / "outputs" / "feature_ablation" / "feature_combinations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMBO_LAYERS = [24, 27, 30, 34]
BEST_SINGLE  = 0.916   # best single-feature val_acc from linear_probe.py
RANDOM_BASE  = 1 / 31

FAMILIES = [
    {
        "name":     "prompt_first · audio_mean",
        "template": "prompt_first_l{L}_audio_mean",
        "c_ind":    "#aec7e8",   # individual layer bars (light)
        "c_mean":   "#1f77b4",   # mean combination (medium)
        "c_cat":    "#0d3b6b",   # concat combination (dark)
    },
    {
        "name":     "audio_first · audio_mean",
        "template": "audio_first_l{L}_audio_mean",
        "c_ind":    "#ffbb78",
        "c_mean":   "#ff7f0e",
        "c_cat":    "#7a3800",
    },
    {
        "name":     "audio_first · last_text",
        "template": "audio_first_l{L}_last_text",
        "c_ind":    "#f4a5a5",
        "c_mean":   "#d62728",
        "c_cat":    "#7a0000",
    },
]

# ── Load features ──────────────────────────────────────────────────────────────
print("Loading features...")
tr = torch.load(FEAT_DIR / "train_20pc_features.pt", weights_only=False)
va = torch.load(FEAT_DIR / "val_10pc_features.pt",   weights_only=False)
y_tr = tr["labels"].numpy()
y_va = va["labels"].numpy()
print(f"  train {len(y_tr)}  |  val {len(y_va)}")


def probe(X_tr, X_va):
    sc  = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(
        sc.transform(X_tr), y_tr
    )
    return clf.score(sc.transform(X_va), y_va)


# ── Run all probes ─────────────────────────────────────────────────────────────
print("\nRunning probes...")
records = []

for fam in FAMILIES:
    print(f"\n  [{fam['name']}]")

    layer_Xtr, layer_Xva = [], []

    for L in COMBO_LAYERS:
        key = fam["template"].format(L=L)
        Xtr = tr["features"][key].float().numpy()
        Xva = va["features"][key].float().numpy()
        layer_Xtr.append(Xtr)
        layer_Xva.append(Xva)

        acc = probe(Xtr, Xva)
        records.append({
            "family": fam["name"], "method": f"L{L}",
            "input_dim": 2048, "val_acc": acc,
        })
        print(f"    L{L:<3}             {acc:.3f}")

    # Mean combination
    acc_mean = probe(np.mean(layer_Xtr, axis=0), np.mean(layer_Xva, axis=0))
    records.append({
        "family": fam["name"], "method": "mean[24,27,30,34]",
        "input_dim": 2048, "val_acc": acc_mean,
    })
    print(f"    mean[24,27,30,34]  {acc_mean:.3f}")

    # Concat combination
    acc_cat = probe(np.concatenate(layer_Xtr, axis=1),
                    np.concatenate(layer_Xva, axis=1))
    records.append({
        "family": fam["name"], "method": "concat[24,27,30,34]",
        "input_dim": 8192, "val_acc": acc_cat,
    })
    print(f"    concat[24,27,30,34]{acc_cat:.3f}")

# ── Save CSV ───────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["family", "method", "input_dim", "val_acc"])
    writer.writeheader()
    writer.writerows(records)
print(f"\nResults saved -> {csv_path}")

# ── Console summary ────────────────────────────────────────────────────────────
print(f"\n{'Family':<33}  {'Method':<22}  {'Dim':<5}  Val Acc")
print("-" * 72)
for r in records:
    marker = " <== best" if r["val_acc"] == max(x["val_acc"] for x in records) else ""
    print(f"  {r['family']:<31}  {r['method']:<22}  {r['input_dim']:<5}  {r['val_acc']:.3f}{marker}")
print(f"\n  Random baseline (31 classes): {RANDOM_BASE:.3f}")
print(f"  Best single feature (linear_probe.py):  {BEST_SINGLE:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
BAR_LABELS = [f"L{L}" for L in COMBO_LAYERS] + ["mean\n[24-34]", "concat\n[24-34]"]
N_PER_FAM  = len(BAR_LABELS)   # 6
BAR_W      = 0.55
GROUP_GAP  = 1.2
N_FAM      = len(FAMILIES)

fig, ax = plt.subplots(figsize=(15, 6))

xtick_pos, xtick_lab = [], []

for fi, fam in enumerate(FAMILIES):
    fam_recs = [r for r in records if r["family"] == fam["name"]]
    base_x   = fi * (N_PER_FAM * BAR_W + GROUP_GAP)

    for bi, (r, lbl) in enumerate(zip(fam_recs, BAR_LABELS)):
        x = base_x + bi * BAR_W

        if r["method"].startswith("concat"):
            color, hatch, edge = fam["c_cat"], "//", fam["c_cat"]
        elif r["method"].startswith("mean"):
            color, hatch, edge = fam["c_mean"], None, fam["c_mean"]
        else:
            color, hatch, edge = fam["c_ind"], None, fam["c_ind"]

        ax.bar(x, r["val_acc"], width=BAR_W * 0.88,
               color=color, hatch=hatch, edgecolor="white", linewidth=0.4)

        xtick_pos.append(x)
        xtick_lab.append(lbl)

    # Group label below x-axis
    group_cx = base_x + (N_PER_FAM - 1) * BAR_W / 2
    ax.text(group_cx, -0.095, fam["name"],
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            transform=ax.get_xaxis_transform())

    # Vertical separator between groups (skip after last)
    if fi < N_FAM - 1:
        sep_x = base_x + N_PER_FAM * BAR_W + GROUP_GAP / 2 - BAR_W / 2
        ax.axvline(sep_x, color="#cccccc", linewidth=1.0)

ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_lab, fontsize=7.5)
ax.set_ylabel("Val Accuracy  (31 intent classes)", fontsize=10)
ax.set_title(
    "Feature Combination Ablation: Single Layer vs Mean vs Concat  ·  Layers [24, 27, 30, 34]\n"
    "Qwen2.5-Omni-3B (frozen, 4-bit)  ·  FSC small ablation  ·  Logistic Regression Probe  "
    "·  train 620 / val 310",
    fontsize=9.5, pad=10,
)
ax.axhline(RANDOM_BASE,  color="gray",  linestyle=":",  linewidth=1.2,
           label=f"random ({RANDOM_BASE:.3f})")
ax.axhline(BEST_SINGLE,  color="black", linestyle="--", linewidth=1.2, alpha=0.45,
           label=f"best single feature ({BEST_SINGLE:.3f})")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3, zorder=0)

# Legend
legend_handles = []
for fam in FAMILIES:
    short = fam["name"]
    legend_handles += [
        mpatches.Patch(color=fam["c_ind"],  label=f"{short}  (individual)"),
        mpatches.Patch(color=fam["c_mean"], label=f"{short}  (mean)"),
        mpatches.Patch(color=fam["c_cat"],  hatch="//", label=f"{short}  (concat)"),
    ]
legend_handles += [
    plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.45,
               label=f"best single ({BEST_SINGLE:.3f})"),
    plt.Line2D([0], [0], color="gray", linestyle=":",
               label=f"random ({RANDOM_BASE:.3f})"),
]
ax.legend(handles=legend_handles, fontsize=6.8, loc="lower right",
          ncol=2, framealpha=0.92)

plt.subplots_adjust(bottom=0.18)
plot_path = OUT_DIR / "feature_combinations.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved  -> {plot_path}")
print("Done.")
