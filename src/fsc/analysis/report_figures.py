"""
Figures for the FSC chapter of the report.

    fig_fsc_feature_selection.png   which teacher representation to distil
    fig_fsc_probe.png               how far the bottleneck can be compressed
    fig_fsc_kd_2x2.png              KD gain under capacity and data limits
    fig_fsc_hubert_comparison.png   multimodal teacher against an audio-only one

Same palette and axis style as the IEMOCAP report figures, so the two chapters
look like they came from the same document. The earlier versions of these plots
were the raw script output, with all 44 features spelled out and the full run
configuration in the title. Settings belong in the caption instead.

Two plots that used to be in the chapter are not produced here. The KD sweep is
a tuning record rather than a result, and the final-test bars only repeat a
table.

Everything is read from the CSVs under outputs/fsc/, so nothing is retrained.

    python src/fsc/analysis/report_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ROOT = _SRC.parent
FSC = ROOT / "outputs" / "fsc"
OUT = FSC / "report"

# shared with common.repr_analysis, restated here so the file runs on its own
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c2367f", "#a8760a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e0"
HIGHLIGHT, MUTED = PALETTE[0], "#8a8983"

METHOD_LABEL = {"ce_only": "CE", "logit_kd": "logit-KD",
                "feature_kd": "feature-KD", "full_kd": "full-KD"}
SETTING_LABEL = {"strong_100": "strong\nfull data", "strong_20": "strong\n20% data",
                 "small_100": "small\nfull data", "small_20": "small\n20% data"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- figure 1
def feature_selection():
    lp = pd.read_csv(FSC / "feature_ablation" / "linear_probe" / "results.csv")
    fc = pd.read_csv(FSC / "feature_ablation" / "feature_combinations" / "results.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) accuracy against layer, one line per family
    families = [
        (("prompt_first", "audio_mean"), "prompt-first, audio mean", HIGHLIGHT, "-", "o", 2.0),
        (("audio_first", "audio_mean"), "audio-first, audio mean", MUTED, "-", "s", 1.4),
        (("audio_first", "last_text"), "audio-first, last text", MUTED, "--", "^", 1.4),
    ]
    for (order, pooling), label, colour, ls, mk, lw in families:
        sub = lp[(lp.order == order) & (lp.pooling == pooling)].sort_values("layer")
        ax1.plot(sub.layer, sub.val_acc, ls, marker=mk, color=colour, linewidth=lw,
                 markersize=4.2, label=label, zorder=3 if colour == HIGHLIGHT else 2)

    ref = lp[(lp.order == "layer0") & (lp.pooling == "audio_mean")].val_acc.iloc[0]
    ax1.axhline(ref, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax1.text(9.4, ref + 0.014, "projected audio, before the LLM blocks",
             fontsize=7.5, color=INK_MUTED)

    ax1.set_xlabel("layer", fontsize=8.5, color=INK_SOFT)
    ax1.set_ylabel("validation accuracy", fontsize=8.5, color=INK_SOFT)
    ax1.set_title("(a) accuracy by layer", fontsize=10, color=INK, loc="left", pad=6)
    ax1.set_xticks([9, 18, 24, 27, 30, 34, 36])
    ax1.set_ylim(0.52, 0.965)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right", labelcolor=INK_SOFT)
    _style(ax1)

    # (b) best single layer against the two ways of combining four layers
    fam_order = ["prompt_first · audio_mean",
                 "audio_first · audio_mean",
                 "audio_first · last_text"]
    fam_short = ["prompt-first\naudio mean", "audio-first\naudio mean",
                 "audio-first\nlast text"]
    singles = ["L24", "L27", "L30", "L34"]
    series = [("best single layer", MUTED),
              ("mean of the four", HIGHLIGHT),
              ("concatenate the four", "#c9c8c2")]

    vals = []
    for fam in fam_order:
        s = fc[fc.family == fam]
        best = s[s.method.isin(singles)].val_acc.max()
        mean = s[s.method.str.startswith("mean")].val_acc.iloc[0]
        cat = s[s.method.str.startswith("concat")].val_acc.iloc[0]
        vals.append((best, mean, cat))
    vals = np.array(vals)

    x = np.arange(len(fam_order))
    w = 0.26
    for i, (label, colour) in enumerate(series):
        off = (i - 1) * w
        ax2.bar(x + off, vals[:, i], width=w * 0.92, color=colour,
                edgecolor="none", label=label, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fam_short, fontsize=8)
    ax2.set_ylim(0.80, 0.952)
    ax2.set_ylabel("validation accuracy", fontsize=8.5, color=INK_SOFT)
    ax2.set_title("(b) combining layers 24, 27, 30 and 34", fontsize=10, color=INK,
                  loc="left", pad=6)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=INK_SOFT)
    _style(ax2)

    # mark only the representation that is actually used downstream
    ax2.annotate("selected", xy=(0, vals[0, 1]), xytext=(0, vals[0, 1] + 0.009),
                 fontsize=7.5, color=HIGHLIGHT, ha="center")

    fig.suptitle("Teacher feature selection on FSC", fontsize=12, color=INK,
                 x=0.007, ha="left", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_fsc_feature_selection.png")


# ---------------------------------------------------------------- figure 2
def probe_compression():
    p = pd.read_csv(FSC / "teacher_probe" / "results.csv").set_index("id")

    # (probe id, tick label) in the order they are drawn
    baseline = [("A1", "none")]
    one_layer = [("B1", "32"), ("B2", "64"), ("B3", "128"), ("B4", "256"),
                 ("A2", "1024"), ("A3", "2048")]
    two_layer = [("C1", "32"), ("C2", "64"), ("C3", "128")]

    groups = [("linear baseline", baseline, [0.0]),
              ("one hidden layer", one_layer, list(np.arange(1.5, 7.5, 1.0))),
              ("two hidden layers, via 512", two_layer, [9.0, 10.0, 11.0])]

    xs, labels, acc, f1 = [], [], [], []
    for _, items, positions in groups:
        for (pid, lab), x in zip(items, positions):
            xs.append(x)
            labels.append(lab)
            acc.append(float(p.loc[pid, "eval_acc"]))
            f1.append(float(p.loc[pid, "eval_macro_f1"]))
    xs = np.array(xs)

    fig, ax = plt.subplots(figsize=(8.8, 3.9), facecolor=SURFACE)
    w = 0.38
    ax.bar(xs - w / 2, acc, width=w * 0.92, color=PALETTE[0], edgecolor="none",
           label="accuracy", zorder=3)
    ax.bar(xs + w / 2, f1, width=w * 0.92, color=PALETTE[1], edgecolor="none",
           label="macro-F1", zorder=3)

    # the linear probe is the thing every other row has to beat
    ax.axhline(acc[0], color=INK_SOFT, linestyle="--", linewidth=1.0, zorder=4)
    ax.text(0.62, acc[0] + 0.0005, "linear probe", fontsize=7.5, color=INK_SOFT,
            ha="left")

    # ring the width that becomes the student's target
    sel = list(labels).index("64")
    ax.annotate("used for the student", xy=(xs[sel], 0.9641), xytext=(xs[sel], 0.9668),
                fontsize=7.6, color=INK_SOFT, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, linewidth=0.8,
                                shrinkA=1, shrinkB=1))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.944, 0.9705)
    ax.set_ylabel("validation score, 31 intent classes", fontsize=8.5, color=INK_SOFT)
    ax.set_xlabel("hidden layer width", fontsize=8.5, color=INK_SOFT, labelpad=30)
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2,
              bbox_to_anchor=(1.0, 1.005), labelcolor=INK_SOFT)
    _style(ax)

    # group names under the tick labels, with a rule spanning each group
    tr = ax.get_xaxis_transform()
    for name, items, positions in groups:
        lo, hi = min(positions), max(positions)
        pad = 0.42 if lo != hi else 0.42
        ax.plot([lo - pad, hi + pad], [-0.105, -0.105], color=GRID, linewidth=1.2,
                transform=tr, clip_on=False, zorder=5)
        ax.text((lo + hi) / 2, -0.135, name, fontsize=8, color=INK_SOFT,
                ha="center", va="top", transform=tr)

    ax.set_title("Teacher probe architectures", fontsize=12, color=INK,
                 loc="left", pad=8)
    fig.tight_layout()
    _save(fig, "fig_fsc_probe.png")


# ---------------------------------------------------------------- figure 3
def kd_2x2():
    runs = pd.read_csv(FSC / "student" / "kd_2x2_tuned" / "results.csv")

    # paired: each seed is compared against the CE run with the same seed
    ce = runs[runs.method == "ce_only"].set_index(["setting", "seed"]).macro_f1
    rows = []
    for (setting, method, seed), g in runs.groupby(["setting", "method", "seed"]):
        if method == "ce_only":
            continue
        rows.append({"setting": setting, "method": method, "seed": seed,
                     "gain": (g.macro_f1.iloc[0] - ce.loc[(setting, seed)]) * 100})
    gains = pd.DataFrame(rows)
    agg = gains.groupby(["setting", "method"]).gain.agg(["mean", "std"]).reset_index()

    settings = ["strong_100", "strong_20", "small_100", "small_20"]
    methods = ["logit_kd", "feature_kd", "full_kd"]
    colours = [HIGHLIGHT, MUTED, PALETTE[3]]

    fig, ax = plt.subplots(figsize=(8.2, 3.8), facecolor=SURFACE)
    x = np.arange(len(settings))
    w = 0.26
    for i, (m, colour) in enumerate(zip(methods, colours)):
        off = (i - 1) * w
        sub = agg[agg.method == m].set_index("setting").loc[settings]
        ax.bar(x + off, sub["mean"], width=w * 0.9, color=colour, edgecolor="none",
               label=METHOD_LABEL[m], zorder=3)
        ax.errorbar(x + off, sub["mean"], yerr=sub["std"], fmt="none",
                    ecolor=INK_SOFT, elinewidth=0.9, capsize=2.5, zorder=4)

    ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABEL[s] for s in settings], fontsize=8.5)
    ax.set_ylabel("validation macro-F1 gain over CE (pp)", fontsize=8.5, color=INK_SOFT)
    ax.set_ylim(0, 9.6)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=3, labelcolor=INK_SOFT)
    _style(ax)
    ax.set_title("Distillation gain under capacity and data limits", fontsize=12,
                 color=INK, loc="left", pad=8)
    fig.tight_layout()
    _save(fig, "fig_fsc_kd_2x2.png")


# ---------------------------------------------------------------- figure 4
def hubert_control():
    c = pd.read_csv(FSC / "hubert_baseline" / "final_test" / "comparison.csv")
    c = c.set_index("method")
    qwen_ce = c.loc["ce_only", "qwen_test_macro_f1"]
    hub_ce = c.loc["ce_only", "hubert_test_macro_f1"]

    methods = ["logit_kd", "feature_kd", "full_kd"]
    qwen = [(c.loc[m, "qwen_test_macro_f1"] - qwen_ce) * 100 for m in methods]
    hub = [(c.loc[m, "hubert_test_macro_f1"] - hub_ce) * 100 for m in methods]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    x = np.arange(len(methods))
    w = 0.3
    ax.bar(x - w / 2, qwen, width=w * 0.9, color=HIGHLIGHT, edgecolor="none",
           label="Qwen2.5-Omni (multimodal)", zorder=3)
    ax.bar(x + w / 2, hub, width=w * 0.9, color=MUTED, edgecolor="none",
           label="HuBERT-large (audio only)", zorder=3)

    ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], fontsize=9)
    ax.set_ylabel("test macro-F1 gain over CE (pp)", fontsize=8.5, color=INK_SOFT)
    ax.set_ylim(0, 3.2)
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK_SOFT)
    _style(ax)
    ax.set_title("Multimodal teacher against an audio-only teacher", fontsize=12,
                 color=INK, loc="left", pad=8)
    fig.tight_layout()
    _save(fig, "fig_fsc_hubert_comparison.png")


def main():
    feature_selection()
    probe_compression()
    kd_2x2()
    hubert_control()


if __name__ == "__main__":
    main()
