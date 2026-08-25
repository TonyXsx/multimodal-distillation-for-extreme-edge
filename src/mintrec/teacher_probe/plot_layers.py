"""
Which pooled layer carries the most intent signal?

Reads the probe results, fixes the head at A2 (the sweet spot from the capacity
sweep) and plots dev acc / macro-F1 per layer, one line per feature set.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_OUTPUTS  # noqa: E402

OUT_DIR = MINTREC_OUTPUTS / "teacher_probe"
HEAD = "A2"
FEAT_ORDER = ["pf_audio_mean_l24", "pf_audio_mean_l27", "pf_audio_mean_l30",
              "pf_audio_mean_l34", "pf_audio_mean_L24-27-30-34"]
XLAB = ["L24", "L27", "L30", "L34", "mean\n24-30-34"]


def vlabel(tag):
    if "pf_text-audio" in tag: return "ta: audio+text"
    if "bf16" in tag:          return "bf16 tva"
    if "aware" in tag:         return "4bit tva-aware"
    return "4bit tva-plain"


def main():
    rows = list(csv.DictReader(open(OUT_DIR / "results.csv", encoding="utf-8")))
    rows = [r for r in rows if r["arch"] == HEAD]
    variants = sorted({r["variant"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in zip(["dev_acc", "dev_macro_f1"], axes, ["dev accuracy", "dev macro-F1"]):
        for tag in variants:
            ys = [float(next(r[metric] for r in rows if r["variant"] == tag and r["feature"] == f))
                  for f in FEAT_ORDER]
            ax.plot(range(len(FEAT_ORDER)), ys, marker="o", label=vlabel(tag))
        ax.set_xticks(range(len(FEAT_ORDER))); ax.set_xticklabels(XLAB, fontsize=9)
        ax.set_title(title); ax.set_ylabel(metric); ax.set_xlabel("pooled layer")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"MIntRec2.0 teacher-probe per-layer (head {HEAD}, dev)", fontsize=11)
    fig.tight_layout()
    png = OUT_DIR / "probe_layers.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")

    # mean across the four feature sets, per layer
    print(f"per-layer mean across {len(variants)} feature sets (head {HEAD}):")
    print(f"{'layer':<16}{'dev_acc':<10}{'macroF1':<10}")
    agg = defaultdict(lambda: [[], []])
    for r in rows:
        agg[r["feature"]][0].append(float(r["dev_acc"]))
        agg[r["feature"]][1].append(float(r["dev_macro_f1"]))
    for f in FEAT_ORDER:
        a, m = agg[f]
        print(f"{f.replace('pf_audio_mean_',''):<16}{sum(a)/len(a):<10.4f}{sum(m)/len(m):<10.4f}")
    print(f"\nPlot -> {png}")


if __name__ == "__main__":
    main()
