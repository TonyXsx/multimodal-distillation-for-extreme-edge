"""
Linear probe ablation over all 44 frozen teacher features.

Trains a logistic regression (no projection, pure linear separability test)
on each feature independently, evaluates on val, and produces:

  outputs/feature_ablation/linear_probe/
    feature_ablation_linear_probe.png   <- ranked bar chart + layer curves
    results.csv                         <- full results table

This script only reads pre-extracted feature files and runs CPU-only sklearn.
No GPU needed. Runtime: < 1 min.

Inputs:
  data/teacher_features/fsc_small_ablation__qwen2.5-omni-3b-4bit/
    train_20pc_features.pt
    val_10pc_features.pt
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT  = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
FEAT_DIR = PROJECT / "data" / "teacher_features" / "fsc_small_ablation__qwen2.5-omni-3b-4bit"
OUT_DIR  = PROJECT / "outputs" / "feature_ablation" / "linear_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LLM_LAYERS = [9, 18, 24, 27, 30, 34, 36]
ALL_LAYERS = [0] + LLM_LAYERS   # 0 = projected audio (before any LLM block)

# ── Feature name parser ────────────────────────────────────────────────────────
def parse_feature(name: str) -> tuple[str, int, str]:
    """Returns (order, layer_idx, pooling_type)."""
    if name.startswith("projected_"):
        return "layer0", 0, name[len("projected_"):]
    if name.startswith("prompt_first_"):
        rest = name[len("prompt_first_"):]
    else:
        rest = name[len("audio_first_"):]
    order = "prompt_first" if name.startswith("prompt_first") else "audio_first"
    layer_str, pooling = rest.split("_", 1)
    return order, int(layer_str[1:]), pooling  # strip 'l' prefix from e.g. 'l27'

# ── Load features ──────────────────────────────────────────────────────────────
print("Loading features...")
tr = torch.load(FEAT_DIR / "train_20pc_features.pt", weights_only=False)
va = torch.load(FEAT_DIR / "val_10pc_features.pt",   weights_only=False)
y_tr = tr["labels"].numpy()
y_va = va["labels"].numpy()
feat_names = list(tr["features"].keys())
print(f"  {len(feat_names)} features  |  train {len(y_tr)}  |  val {len(y_va)}")

# ── Run logistic regression for every feature ──────────────────────────────────
print("\nRunning linear probes...")
records = []
for name in feat_names:
    order, layer, pooling = parse_feature(name)
    X_tr = tr["features"][name].float().numpy()
    X_va = va["features"][name].float().numpy()

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs"
    ).fit(scaler.transform(X_tr), y_tr)
    acc = clf.score(scaler.transform(X_va), y_va)

    records.append({"feature": name, "order": order, "layer": layer,
                    "pooling": pooling, "val_acc": acc})
    print(f"  {name:<42}  {acc:.3f}")

records.sort(key=lambda r: r["val_acc"], reverse=True)

# ── Save CSV ────────────────────────────────────────────────────────────────────
import csv
csv_path = OUT_DIR / "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["rank","feature","order","layer","pooling","val_acc"])
    writer.writeheader()
    for i, r in enumerate(records, 1):
        writer.writerow({"rank": i, **r})
print(f"\nResults saved -> {csv_path}")

print(f"\n{'Rank':<5} {'val_acc':<8} {'feature'}")
print("-" * 60)
for i, r in enumerate(records, 1):
    print(f"  {i:<3}  {r['val_acc']:.3f}   {r['feature']}")
print(f"\nRandom baseline (31 classes): {1/31:.3f}")

# ── Colour / style maps ────────────────────────────────────────────────────────
COLOR = {
    ("layer0",       "audio_mean"):        "#888888",
    ("layer0",       "audio_last"):        "#cccccc",
    ("prompt_first", "audio_mean"):        "#1f77b4",
    ("prompt_first", "audio_last"):        "#7fbfdb",
    ("audio_first",  "audio_mean"):        "#ff7f0e",
    ("audio_first",  "audio_last"):        "#ffbc6f",
    ("audio_first",  "last_text"):         "#d62728",
    ("audio_first",  "last_4_text_mean"):  "#f4a5a5",
}
LEGEND = {
    ("layer0",       "audio_mean"):        "layer0 · projected_audio_mean",
    ("layer0",       "audio_last"):        "layer0 · projected_audio_last",
    ("prompt_first", "audio_mean"):        "prompt_first · audio_mean",
    ("prompt_first", "audio_last"):        "prompt_first · audio_last",
    ("audio_first",  "audio_mean"):        "audio_first  · audio_mean",
    ("audio_first",  "audio_last"):        "audio_first  · audio_last",
    ("audio_first",  "last_text"):         "audio_first  · last_text",
    ("audio_first",  "last_4_text_mean"):  "audio_first  · last_4_text_mean",
}
LINE = {
    "audio_mean":       ("-",   "o"),
    "audio_last":       ("--",  "s"),
    "last_text":        ("-",   "^"),
    "last_4_text_mean": (":",   "D"),
}

def lookup(records, order, layer, pooling):
    for r in records:
        if r["order"] == order and r["layer"] == layer and r["pooling"] == pooling:
            return r["val_acc"]
    return None

# ── Build figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.0, 1.0], wspace=0.35)

ax_bar = fig.add_subplot(gs[0])
gs_r   = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.5)
ax_pf  = fig.add_subplot(gs_r[0])
ax_af  = fig.add_subplot(gs_r[1])

# ── Left: ranked horizontal bar chart ─────────────────────────────────────────
sorted_asc = list(reversed(records))          # ascending so best feature is at the top
bar_colors = [COLOR.get((r["order"], r["pooling"]), "#aaaaaa") for r in sorted_asc]
bar_accs   = [r["val_acc"] for r in sorted_asc]
bar_labels = [r["feature"] for r in sorted_asc]
y_pos      = np.arange(len(sorted_asc))

ax_bar.barh(y_pos, bar_accs, color=bar_colors, edgecolor="none", height=0.72)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(bar_labels, fontsize=7.8)
ax_bar.set_xlabel("Logistic Regression Val Accuracy  (31 intent classes)", fontsize=10)
ax_bar.set_title(
    "All 44 Teacher Features — Ranked by Linear Probe Val Accuracy\n"
    "Qwen2.5-Omni-3B (frozen, 4-bit)  ·  FSC small ablation  ·  train 620 / val 310",
    fontsize=10, pad=10,
)
ax_bar.axvline(1 / 31, color="gray", linestyle=":", linewidth=1.2, label=f"random ({1/31:.3f})")
ax_bar.set_xlim(0, 1.05)
ax_bar.grid(axis="x", alpha=0.3)

legend_patches = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR[k], label=LEGEND[k])
    for k in LEGEND
]
ax_bar.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color="gray", linestyle=":", label=f"random ({1/31:.3f})")
], fontsize=7.5, loc="lower right", framealpha=0.9)

# ── Top-right: prompt_first layer curves (+ layer-0 reference) ────────────────
for pooling, (ls, mk) in LINE.items():
    if pooling not in ("audio_mean", "audio_last"):
        continue
    key = ("prompt_first", pooling)
    xs = [L for L in LLM_LAYERS]
    ys = [lookup(records, "prompt_first", L, pooling) for L in LLM_LAYERS]
    ax_pf.plot(xs, ys, linestyle=ls, marker=mk, color=COLOR[key],
               label=pooling, linewidth=1.8, markersize=5)

# layer-0 projected reference (stars at x=0)
for pooling in ("audio_mean", "audio_last"):
    v = lookup(records, "layer0", 0, pooling)
    if v is not None:
        ax_pf.scatter([0], [v], color=COLOR[("layer0", pooling)],
                      marker="*", s=100, zorder=6, label=f"layer0·{pooling}")

ax_pf.axhline(1/31, color="gray", linestyle=":", linewidth=1)
ax_pf.set_title("prompt_first · accuracy vs LLM layer", fontsize=9)
ax_pf.set_xlabel("Layer index  (0 = projected, before LLM)", fontsize=8)
ax_pf.set_ylabel("Val accuracy", fontsize=8)
ax_pf.set_xticks([0] + LLM_LAYERS)
ax_pf.set_xticklabels(["0\n(proj)"] + [str(l) for l in LLM_LAYERS], fontsize=7.5)
ax_pf.set_ylim(0, 1.05)
ax_pf.legend(fontsize=7.5, loc="lower right")
ax_pf.grid(alpha=0.3)

# ── Bottom-right: audio_first layer curves ────────────────────────────────────
for pooling, (ls, mk) in LINE.items():
    key = ("audio_first", pooling)
    xs  = [L for L in LLM_LAYERS]
    ys  = [lookup(records, "audio_first", L, pooling) for L in LLM_LAYERS]
    ax_af.plot(xs, ys, linestyle=ls, marker=mk, color=COLOR[key],
               label=pooling, linewidth=1.8, markersize=5)

ax_af.axhline(1/31, color="gray", linestyle=":", linewidth=1)
ax_af.set_title("audio_first · accuracy vs LLM layer", fontsize=9)
ax_af.set_xlabel("Layer index", fontsize=8)
ax_af.set_ylabel("Val accuracy", fontsize=8)
ax_af.set_xticks(LLM_LAYERS)
ax_af.set_xticklabels([str(l) for l in LLM_LAYERS], fontsize=7.5)
ax_af.set_ylim(0, 1.05)
ax_af.legend(fontsize=7.5, loc="lower right")
ax_af.grid(alpha=0.3)

# ── Save ───────────────────────────────────────────────────────────────────────
plot_path = OUT_DIR / "feature_ablation_linear_probe.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved  -> {plot_path}")
print("Done.")
