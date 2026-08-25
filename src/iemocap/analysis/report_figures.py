"""
The figures for the report. Four of them, each answering one question the
numbers alone cannot.

    fig_loso_five_fold.png                 the headline: 5-fold leave-one-session-out
    fig_results_delta_vs_ce.png            the two single splits, SI and SD
    fig_mechanism_target_fidelity.png      WHY joint Feature-KD does nothing
    fig_repr_{si,sd}_emotion_and_speaker.png   what the embedding actually holds
    fig_teacher_bottleneck_2048_vs_64.png  whether the 64-d target is a good one

Everything is the AUDIO target (`audio_mean_l27`) and, where a stage-2 head is
needed, `mlp_kd`. The last_token target scores the same (test difference 0.38pp,
p = 0.485) but it has seen the transcript, which an audio-only student cannot
reach; audio keeps the story consistent from teacher to student, and it is the
variant val selects in both protocols.

Unlike the CSVs, the t-SNE panels show ONE seed -- the best-on-test of the five,
per encoder. A 2-D embedding of five different runs cannot be averaged, and the
panels are there to show the shape of the solution, not to measure it. Every
number quoted in the report comes from the 5-seed tables instead.

This script reads both protocols in one process, so it builds paths explicitly
rather than through iemocap.paths, whose PROTOCOL is fixed at import.

Outputs: outputs/iemocap/analysis/report/

Usage:
    python src/iemocap/analysis/report_figures.py
    python src/iemocap/analysis/report_figures.py --embed pca      # quick draft
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import normalize

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.probe import Probe  # noqa: E402
from common.repr_analysis import (  # noqa: E402
    INK, INK_MUTED, INK_SOFT, PALETTE, SURFACE, _style, cosine_separability,
    embed_2d, knn_transfer,
)

ROOT = _SRC.parent
DATA = ROOT / "data" / "iemocap"
OUT = ROOT / "outputs" / "iemocap" / "analysis" / "report"
STUDENT_CSV = ROOT / "outputs" / "iemocap" / "student"
TAG = "iemocap4__qwen2.5-omni-3b-bf16-LORA__adapter_ep3__audio-tr"
KEY = "audio_mean_l27"
CLASSES = ["angry", "happy", "neutral", "sad"]
PROTOCOLS = ("si", "sd")
SPK_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c2367f",
               "#a8760a", "#00868b", "#8b2f5f", "#5b7c1f", "#9a4a1f"]

# Figures carry no prose. Panel labels are (a)/(b), the axes are labelled, and
# everything a reader needs beyond that belongs in the caption -- which is how
# these get typeset in the report anyway. FIG_CAPTIONS below holds the text to
# paste under each one.
PAGE = "#ffffff"
HIGHLIGHT, MUTED = PALETTE[0], "#8a8983"

# label -> (encoder in the CSVs, stage-2 readout or None for the joint head)
METHODS = [
    ("2-stage, audio + mlp_kd",     "feature_only_audio", "mlp_kd"),
    ("2-stage, last_token + mlp_kd", "feature_only",      "mlp_kd"),
    ("logit-KD (T=2)",              "logit_kd",           None),
    ("full-KD, audio (T=2)",        "full_kd_audio",      None),
    ("full-KD, last_token (T=2)",   "full_kd_lasttoken",  None),
    ("feature-KD, audio",           "feature_kd_audio",   None),
    ("feature-KD, last_token",      "feature_kd",         None),
]


def suffix(p):
    return "" if p == "si" else "_sd"


def teacher_test(protocol, key=None):
    """2048-d standardised feature and its 64-d bottleneck, test split."""
    key = key or KEY
    ck = torch.load(DATA / f"teacher_probe{suffix(protocol)}" / "bottleneck" / "adapted"
                    / key / "checkpoint.pt", weights_only=False)
    probe = Probe(ck["in_dim"], [ck["bottleneck"]], len(ck["classes"]), dropout=0.0)
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    root = DATA / f"teacher_features{suffix(protocol)}" / TAG
    out = {}
    with torch.no_grad():
        for split in ("train", "test"):
            r = torch.load(root / f"{split}_features.pt", weights_only=False)
            X = (r["features"][key].float() - ck["mu"]) / ck["sd"]
            out[split] = (X.numpy(), probe(X, return_bottleneck=True)[1].numpy(),
                          r["labels"].numpy(), list(r["sample_ids"]))
    return out


def speakers(protocol, ids):
    man = pd.read_csv(DATA / f"manifest{suffix(protocol)}.csv").set_index("turn_id")
    names = sorted(man.speaker.unique())
    lut = {s: i for i, s in enumerate(names)}
    return np.array([lut[man.loc[i, "speaker"]] for i in ids]), names


def series(runs, st2, protocol, enc, readout, col="test_ua"):
    if readout is None:
        g = runs[(runs.protocol == protocol) & (runs.method == enc)]
    else:
        g = st2[(st2.protocol == protocol) & (st2.method == enc) & (st2.readout == readout)]
    return g.sort_values("seed")[["seed", col]].values


def fig_loso():
    """Five-fold LOSO on its own. Two things have to be visible at once: the
    effect size with its interval, and the fact that it holds in every fold --
    a mean of +2.4pp means something quite different if the five folds are
    +2.1..+2.9 than if they are -2..+7. So each fold is plotted as its own dot
    behind the summary marker, and the right panel shows why fold-level pairing
    is the correct test: the held-out sessions differ by 5pp in difficulty, and
    every one of them still improves."""
    import matplotlib.pyplot as plt
    per = pd.read_csv(STUDENT_CSV / "loso_per_fold.csv")
    tab = pd.read_csv(STUDENT_CSV / "loso_main_table.csv")
    tab = tab[tab.method != "CE (end-to-end)"].sort_values("delta_pp")
    order = tab.method.tolist()

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), facecolor=PAGE,
        gridspec_kw={"width_ratios": [2.7, 1.0], "wspace": 0.28})

    for i, m in enumerate(order):
        row = tab[tab.method == m].iloc[0]
        d = per[per.method == m].delta_pp.values
        hit = row.folds_positive == 5
        colour = PALETTE[0] if hit else INK_MUTED
        # the five folds, jittered so coincident values stay countable
        ax.scatter(d, np.full(len(d), i) + np.linspace(-0.16, 0.16, len(d)),
                   s=26, color=colour, alpha=0.38, linewidths=0, zorder=2)
        ax.errorbar(row.delta_pp, i,
                    xerr=[[row.delta_pp - row.ci95_lo_pp], [row.ci95_hi_pp - row.delta_pp]],
                    fmt="o", ms=8, capsize=4, color=colour, ecolor=colour,
                    elinewidth=2.0, zorder=3)
        ptxt = "p<0.001" if row.p < 0.001 else f"p={row.p:.3f}"
        ax.annotate(f"{int(row.folds_positive)}/5   {ptxt}",
                    (1.005, i), xycoords=("axes fraction", "data"), fontsize=8.5,
                    color=INK if hit else INK_SOFT, va="center",
                    fontweight="bold" if hit else "normal")

    ax.axvline(0, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10, color=INK)
    ax.set_xlabel("test UA improvement over end-to-end CE (pp)", fontsize=9.5, color=INK_SOFT)
    ax.set_title("(a) improvement over CE, by method", fontsize=10, color=INK,
                 loc="left", pad=8)
    _style(ax)
    ax.grid(axis="y", visible=False)

    # right: the paired structure the test exploits
    ce = per[per.method == "CE (end-to-end)"].set_index("fold").test_ua
    best = per[per.method == order[-1]].set_index("fold").test_ua
    folds = list(ce.index)
    for f in folds:
        bx.plot([0, 1], [ce[f], best[f]], color=PALETTE[0], lw=1.6, alpha=0.75,
                marker="o", ms=5)
        bx.annotate(f.replace("loso", "S"), (1.03, best[f]), fontsize=8.5,
                    color=INK_SOFT, va="center")
    bx.set_xlim(-0.25, 1.35)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["CE", "2-stage\naudio + mlp_kd"], fontsize=9.5, color=INK)
    # ticks and label on the right: panel (a)'s p-value column runs into the
    # gutter, so a left-hand y-axis here would sit on top of it
    bx.yaxis.tick_right()
    bx.yaxis.set_label_position("right")
    bx.set_ylabel("test UA", fontsize=9.5, labelpad=6)
    bx.set_title("(b) per-fold pairing", fontsize=10, color=INK, loc="left", pad=8)
    _style(bx)
    bx.grid(axis="x", visible=False)

    fig.suptitle("5-fold leave-one-session-out", fontsize=11.5, color=INK, y=0.99)
    fig.subplots_adjust(left=0.19, right=0.93, top=0.86, bottom=0.14, wspace=0.30)
    fig.savefig(OUT / "fig_loso_five_fold.png", dpi=170, facecolor=PAGE,
                bbox_inches="tight")
    plt.close(fig)
    print("  fig_loso_five_fold.png")


BAR_METHODS = {
    "audio": [("feature-KD", "feature_kd_audio", None),
              ("full-KD", "full_kd_audio", None),
              ("logit-KD", "logit_kd", None),
              ("two-stage", "feature_only_audio", "mlp_kd")],
    "lasttoken": [("feature-KD", "feature_kd", None),
                  ("full-KD", "full_kd_lasttoken", None),
                  ("logit-KD", "logit_kd", None),
                  ("two-stage", "feature_only", "mlp_kd")],
}


def fig_delta_bars(runs, st2, target):
    """Improvement over CE, as bars. Zero is a real quantity here -- it is the CE
    baseline trained on the same data -- so bars from zero are honest, and they
    read faster than a forest plot. logit-KD has no feature target and appears in
    both versions as the reference KD baseline."""
    import matplotlib.pyplot as plt
    meths = BAR_METHODS[target]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True, facecolor=PAGE)
    for ax, proto, tag in zip(axes, ("sd", "si"), ("a", "b")):
        ce = runs[(runs.protocol == proto) & (runs.method == "ce")].sort_values("seed").test_ua.values
        x = np.arange(len(meths), dtype=float)
        for i, (label, enc, ro) in enumerate(meths):
            v = series(runs, st2, proto, enc, ro)[:, 1].astype(float)
            d = (v - ce) * 100
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            ax.bar(i, d.mean(), width=0.62, color=HIGHLIGHT if ro else MUTED,
                   linewidth=0, zorder=2)
            ax.errorbar(i, d.mean(), yerr=half, fmt="none", ecolor="#3a3a38",
                        elinewidth=1.1, capsize=3, zorder=3)
        ax.axhline(0, color="#3a3a38", lw=1.0, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in meths], fontsize=9, rotation=20, ha="right")
        ax.set_title(f"({tag}) {proto.upper()} protocol", fontsize=10, color=INK)
        _style(ax)
        ax.set_facecolor(PAGE)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(r"$\Delta$ test UA vs CE (pp)", fontsize=9.5)
    nice = "audio" if target == "audio" else "last-token"
    fig.suptitle(f"Improvement over CE, {nice} target", fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    name = f"fig_delta_bars_{target}.png"
    fig.savefig(OUT / name, dpi=200, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def fig_mechanism(enc_csv):
    import matplotlib.pyplot as plt
    show = [("feature-KD\n(joint CE + cos)", "feature_kd_audio"),
            ("full-KD\n(CE + logit + cos)", "full_kd_audio"),
            ("two-stage\n(cos alone)", "feature_only_audio")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True, facecolor=PAGE)
    for ax, proto, tag in zip(axes, ("sd", "si"), ("a", "b")):
        d = enc_csv[proto]
        x = np.arange(len(show), dtype=float)
        for j, (split, colour) in enumerate((("train", PALETTE[3]), ("test", PALETTE[1]))):
            vals = [d[d.encoder == e][f"fit_audio_{split}_pct"].mean() for _, e in show]
            off = (j - 0.5) * 0.36
            ax.bar(x + off, vals, width=0.33, color=colour, linewidth=0,
                   label=f"on {split}")
            for xi, v in zip(x + off, vals):
                ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points",
                            xytext=(0, 3 if v >= 0 else -11), ha="center",
                            fontsize=8.5, color=INK_SOFT)
        ax.axhline(0, color=INK_MUTED, lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([s for s, _ in show], fontsize=9, color=INK)
        ax.set_title(f"({tag}) {proto.upper()} protocol", fontsize=10, color=INK)
        _style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("target fidelity (%)", fontsize=9.5)
    fig.suptitle("How much of the teacher's vector the student reproduces",
                 fontsize=11.5, color=INK)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism_target_fidelity.png", dpi=170, facecolor=PAGE,
                bbox_inches="tight")
    plt.close(fig)
    print("  fig_mechanism_target_fidelity.png")


def fig_repr(protocol, runs, st2, method_embed):
    import matplotlib.pyplot as plt
    show = [("CE (no teacher)", "ce", None),
            ("feature-KD (joint)", "feature_kd_audio", None),
            ("two-stage (cos alone)", "feature_only_audio", "mlp_kd")]
    zc = DATA / f"student{suffix(protocol)}" / "z_cache"
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), facecolor=PAGE)
    chance = None
    for j, (label, enc, ro) in enumerate(show):
        s = series(runs, st2, protocol, enc, ro)
        best_seed = int(s[np.argmax(s[:, 1].astype(float)), 0])
        best_ua = float(s[:, 1].astype(float).max())
        f = next(zc.glob(f"{enc}_seed{best_seed}_*.pt"))
        d = torch.load(f, weights_only=False)
        Z, y = d["test"]["z"], d["test"]["y"]
        spk, names = speakers(protocol, method_embed[protocol]["ids"])
        chance = 1.0 / len(np.unique(spk))
        Z2 = embed_2d(Z, method=method_embed["how"], seed=0)
        for row, (what, groups, pal) in enumerate(
                [("by emotion", y, PALETTE[:4]),
                 ("by speaker", spk, SPK_COLOURS)]):
            ax = axes[row, j]
            for i in np.unique(groups):
                m = groups == i
                ax.scatter(Z2[m, 0], Z2[m, 1], s=6, c=pal[i % len(pal)], linewidths=0,
                           alpha=0.75)
            ax.set_xticklabels([]), ax.set_yticklabels([])
            _style(ax)
            if row == 0:
                ax.set_title(f"{label}\nUA {best_ua:.3f}",
                             fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(what, fontsize=10, color=INK_SOFT)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=PALETTE[i], label=c)
               for i, c in enumerate(CLASSES)]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, labelcolor=INK_SOFT,
               ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.012),
               title="emotion (top row); lower row is coloured by speaker",
               title_fontsize=8.5)
    fig.suptitle(f"Student embeddings, {protocol.upper()} test split "
                 f"({len(np.unique(spk))} speakers)", fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(OUT / f"fig_repr_{protocol}_emotion_and_speaker.png", dpi=165,
                facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_repr_{protocol}_emotion_and_speaker.png  "
          f"(speakers in this test split: {len(np.unique(spk))} of {len(names)})")


def fig_bottleneck(teach, how, key):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.4), facecolor=PAGE)
    for r, proto in enumerate(PROTOCOLS):
        hi, lo, y, _ = teach[proto]["test"]
        htr, ltr, ytr, _ = teach[proto]["train"]
        for c, (dim, Z, Ztr) in enumerate((("2048-d hidden state", hi, htr),
                                           ("64-d probe bottleneck", lo, ltr))):
            ax = axes[r, c]
            Z2 = embed_2d(Z, method=how, seed=0)
            for i, cname in enumerate(CLASSES):
                m = y == i
                ax.scatter(Z2[m, 0], Z2[m, 1], s=7, c=PALETTE[i], linewidths=0,
                           alpha=0.8, label=cname if (r == 0 and c == 0) else None)
            gap = cosine_separability(Z, y)["gap"]
            ua = knn_transfer(Ztr, ytr, Z, y, k=10)["ua"]
            ax.set_xticklabels([]), ax.set_yticklabels([])
            _style(ax)
            ax.set_title(f"({chr(97 + r * 2 + c)}) {proto.upper()}, {dim}",
                         fontsize=10, color=INK)
            ax.annotate(f"gap {gap:.3f}\nk-NN UA {ua:.3f}",
                        (0.03, 0.03), xycoords="axes fraction", fontsize=9,
                        color=INK_SOFT, va="bottom")
    fig.legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT, ncols=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.015))
    nice = "audio" if key.startswith("audio") else "last-token"
    tag = "audio" if key.startswith("audio") else "lasttoken"
    fig.suptitle(f"Teacher features before and after the bottleneck ({nice})",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    fig.savefig(OUT / f"fig_teacher_bottleneck_{tag}.png", dpi=170, facecolor=PAGE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_teacher_bottleneck_{tag}.png")


def main():
    ap = argparse.ArgumentParser(description="Report figures, both protocols, audio target.")
    ap.add_argument("--embed", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["loso", "results", "mechanism", "repr", "bottleneck"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {"loso", "results", "mechanism",
                                             "repr", "bottleneck"}

    runs = pd.read_csv(STUDENT_CSV / "fixed_protocol_runs.csv")
    st2 = pd.read_csv(STUDENT_CSV / "stage2_readout.csv")
    enc_csv = {p: pd.read_csv(ROOT / "outputs" / "iemocap" / "analysis" / p
                              / f"{p}_student_encodes_emotion_vs_speaker.csv")
               for p in PROTOCOLS}
    print("figures ->", OUT)

    if "loso" in want:
        fig_loso()
    if "results" in want:
        for tgt in ("audio", "lasttoken"):
            fig_delta_bars(runs, st2, tgt)
    if "mechanism" in want:
        fig_mechanism(enc_csv)
    if {"repr", "bottleneck"} & want:
        teach = {p: teacher_test(p) for p in PROTOCOLS}
        if "bottleneck" in want:
            for k in ("audio_mean_l27", "last_token"):
                fig_bottleneck({p: teacher_test(p, k) for p in PROTOCOLS}, args.embed, k)
        if "repr" in want:
            me = {"how": args.embed,
                  **{p: {"ids": teach[p]["test"][3]} for p in PROTOCOLS}}
            for p in PROTOCOLS:
                fig_repr(p, runs, st2, me)


if __name__ == "__main__":
    main()
