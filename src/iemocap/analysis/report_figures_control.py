"""
Figures for the HuBERT control and the diagnosis that followed it.

    fig_iemocap_target_geometry.png   what the two 64-d targets look like
    fig_iemocap_channels.png          which channel the teachers differ on
    fig_iemocap_pca_folds.png         the unsupervised target, fold by fold

Same style as report_figures.py. Deltas are against the CE end-to-end baseline
used in Table tab:iemocap_loso, seeds averaged within a fold and folds paired,
so every number here lines up with the ones already in the chapter.

The scatter panels use fold 1 and the coordinates saved in
loso1_target_coords.csv, which are the first two principal components of each
target fitted on that fold's training split. A 2-d projection cannot be averaged
over folds; the numbers quoted in the text come from the five-fold tables.

    python src/iemocap/analysis/report_figures_control.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.repr_analysis import INK, INK_SOFT, PALETTE, _style  # noqa: E402

ROOT = _SRC.parent
STUDENT = ROOT / "outputs" / "iemocap" / "student"
COMPARE = ROOT / "outputs" / "iemocap" / "analysis" / "teacher_compare"
OUT = ROOT / "outputs" / "iemocap" / "analysis" / "report"
PAGE = "#ffffff"
CLASSES = ["angry", "happy", "neutral", "sad"]
QWEN, HUBERT = PALETTE[0], PALETTE[1]


def per_fold(csv, method, readout):
    d = pd.read_csv(STUDENT / csv)
    d = d[d.protocol.str.startswith("loso") & (d.method == method) & (d.readout == readout)]
    return d.groupby("protocol").test_ua.mean().sort_index()


def ce_baseline():
    d = pd.read_csv(STUDENT / "fixed_protocol_runs.csv")
    d = d[d.protocol.str.startswith("loso") & (d.method == "ce")]
    return d.groupby("protocol").test_ua.mean().sort_index()


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def cosine_gap(g):
    z, y = unit(g[["pc1", "pc2"]].values), g.y.values
    same = np.zeros((len(y), len(y)), bool)
    for c in np.unique(y):
        same |= np.outer(y == c, y == c)
    S = z @ z.T
    off = ~np.eye(len(y), dtype=bool)
    return S[same & off].mean() - S[(~same) & off].mean()


def fig_geometry():
    """the label-trained probe against an unsupervised projection of the same
    feature, on both splits. Reading across a row shows what the probe does to
    the geometry; reading down a column compares the teachers."""
    import matplotlib.pyplot as plt
    d = pd.read_csv(COMPARE / "loso1_target_coords.csv")
    cols = [("pca64", "train"), ("pca64", "test"), ("probe64", "train"), ("probe64", "test")]
    names = ["PCA-64, train", "PCA-64, test", "probe-64, train", "probe-64, test"]
    fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.9), facecolor=PAGE)
    for r, teacher in enumerate(("qwen", "hubert")):
        for c, ((rep, split), name) in enumerate(zip(cols, names)):
            ax = axes[r, c]
            g = d[(d.teacher == teacher) & (d.rep == rep) & (d.split == split)]
            for i in range(4):
                s = g[g.y == i]
                ax.scatter(s.pc1, s.pc2, s=3.0, c=PALETTE[i], alpha=0.42, linewidths=0,
                           rasterized=True, label=CLASSES[i] if (r == 0 and c == 0) else None)
            for i in range(4):
                s = g[g.y == i]
                ax.scatter(s.pc1.mean(), s.pc2.mean(), s=110, c=PALETTE[i], marker="X",
                           edgecolor="white", linewidths=1.4, zorder=5)
            ax.set_title(f"{name}   gap {cosine_gap(g):.2f}", fontsize=9, color=INK, pad=5)
            _style(ax)
            ax.set_facecolor(PAGE)
            ax.grid(visible=False)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel("Qwen" if teacher == "qwen" else "HuBERT",
                              fontsize=10.5, color=INK, labelpad=8)
    fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.0), markerscale=3.5)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(OUT / "fig_iemocap_target_geometry.png", dpi=180, facecolor=PAGE,
                bbox_inches="tight")
    plt.close(fig)
    print("  fig_iemocap_target_geometry.png")


def fig_channels():
    """the same encoders read out with and without soft labels, so the feature
    term and the logit term can be priced separately."""
    import matplotlib.pyplot as plt
    ce = ce_baseline()
    arms = [("Feature only\n+ label head", "feature_only_audio", "linear"),
            ("Feature only\n+ KD head", "feature_only_audio", "mlp_kd"),
            ("Logit-KD\n+ label head", "logit_kd", "linear"),
            ("Logit-KD\n+ KD head", "logit_kd", "mlp_kd")]
    fig, ax = plt.subplots(figsize=(7.6, 3.7), facecolor=PAGE)
    w = 0.36
    for j, (teacher, csv, colour) in enumerate((("Qwen", "stage2_readout.csv", QWEN),
                                                ("HuBERT", "stage2_readout_hubert.csv", HUBERT))):
        for i, (_, m, r) in enumerate(arms):
            v = (per_fold(csv, m, r).values - ce.values) * 100
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ax.bar(i + (j - 0.5) * w, v.mean(), width=w, color=colour, linewidth=0, zorder=2,
                   label=teacher if i == 0 else None)
            ax.errorbar(i + (j - 0.5) * w, v.mean(), yerr=half, fmt="none", ecolor="#3a3a38",
                        elinewidth=1.0, capsize=3, zorder=3)
    ax.axhline(0, color="#3a3a38", lw=1.0, zorder=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_ylabel(r"$\Delta$ test UA vs CE (pp)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style(ax)
    ax.set_facecolor(PAGE)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_iemocap_channels.png", dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print("  fig_iemocap_channels.png")


def fig_pca_folds():
    """PCA-64 minus probe-64, one dot per fold, against the arm the chapter
    reports."""
    import matplotlib.pyplot as plt
    lc = pd.read_csv(STUDENT / "logit_channel.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.9), facecolor=PAGE)
    for i, (teacher, csv, colour) in enumerate((("HuBERT", "stage2_readout_hubert.csv", HUBERT),
                                                ("Qwen", "stage2_readout.csv", QWEN))):
        a = lc[(lc.teacher == teacher.lower()) & (lc.encoder == "pca64")
               & (lc.readout == "qwen_T2")].groupby("protocol").test_ua.mean().sort_index()
        b = per_fold(csv, "feature_only_audio", "mlp_kd")
        d = (a.values - b.values) * 100
        ax.scatter(d, np.full(len(d), i) + np.linspace(-0.10, 0.10, len(d)), s=42,
                   color=colour, alpha=0.55, linewidths=0, zorder=2)
        half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        ax.errorbar(d.mean(), i, xerr=half, fmt="o", ms=8, capsize=4, color=colour,
                    ecolor=colour, elinewidth=2.0, zorder=3)
        p = stats.ttest_rel(a.values, b.values).pvalue
        ax.annotate(f"{int((d > 0).sum())}/5   p={p:.3f}", (1.005, i),
                    xycoords=("axes fraction", "data"), fontsize=8.5, color=INK_SOFT,
                    va="center")
        for f, v in zip(a.index, d):
            if v < -3:
                ax.annotate(f.replace("loso", "fold "), (v, i + 0.17), fontsize=8,
                            color=INK_SOFT, ha="center")
    ax.axvline(0, color=INK_SOFT, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HuBERT", "Qwen"], fontsize=10, color=INK)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(r"$\Delta$ test UA, unsupervised target minus probe (pp)", fontsize=9.5)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_iemocap_pca_folds.png", dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print("  fig_iemocap_pca_folds.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing figures")
    fig_geometry()
    fig_channels()
    fig_pca_folds()


if __name__ == "__main__":
    main()
