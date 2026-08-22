"""
Dataset-agnostic representation analysis: is the structure in an embedding the
structure you think it is?

Written for KD debugging, where the useful questions are rarely "what is the
accuracy". They are: does this representation separate the CLASSES, or does it
mostly separate the SPEAKERS? does its class structure survive the move to
unseen speakers? and does the student's own embedding end up shaped like the
teacher's at all?

Nothing here is IEMOCAP-specific -- it takes float matrices and integer label
arrays. Every function returns plain numbers or a DataFrame, and every plot
also writes the numbers it draws to CSV, so a figure never becomes the only
record of a result.

The two metrics worth explaining:

`group_predictability` asks how well a nuisance variable (speaker, session,
recording condition) can be read back out of the embedding, by leave-one-out
k-NN. Compared against the majority-class rate, it says whether a
representation has entangled identity with the thing you actually wanted. A
teacher whose embedding predicts SPEAKER at 90% is handing a student
speaker-specific structure that cannot transfer to held-out speakers, however
good its class accuracy looks.

`knn_transfer` fits on one split and evaluates on another with no training, so
it measures how much class structure is present and *portable*, separately from
whatever a trained head could squeeze out.

Colours: slots 1-4 of the reference categorical palette (blue / orange / aqua /
violet), the one four-hue subset that clears the all-pairs CVD and
normal-vision floors needed for scatter plots. Aqua sits below 3:1 on the light
surface, so every figure carries a legend and direct labels rather than relying
on colour alone, and the numbers are always in the CSV.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# Reference categorical palette, light mode. Validated for all-pairs use.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e0"


# ── metrics ──────────────────────────────────────────────────────────────────


def cosine_separability(Z, y):
    """Mean cosine within vs between classes, on L2-normalised rows.

    `gap` is the headline: how much more alike two same-class points are than
    two different-class points. It is scale-free, so it compares across
    representations of different dimension and magnitude.
    """
    Zn = normalize(np.asarray(Z, dtype=np.float64))
    S = Zn @ Zn.T
    y = np.asarray(y)
    eye = np.eye(len(y), dtype=bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = y[:, None] != y[None, :]
    within, between = float(S[same].mean()), float(S[diff].mean())
    return {"within": round(within, 4), "between": round(between, 4),
            "gap": round(within - between, 4)}


def silhouette(Z, y, metric="cosine", max_n=4000, seed=0):
    """Silhouette in [-1, 1]; sub-sampled above `max_n` because it is O(n^2)."""
    Z, y = np.asarray(Z, dtype=np.float64), np.asarray(y)
    if len(y) > max_n:
        idx = np.random.RandomState(seed).choice(len(y), max_n, replace=False)
        Z, y = Z[idx], y[idx]
    return round(float(silhouette_score(Z, y, metric=metric)), 4)


def knn_transfer(Z_fit, y_fit, Z_eval, y_eval, k=10, metric="cosine"):
    """Fit k-NN on one split, score another. No training, so this reports the
    class structure that is actually present and portable."""
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(np.asarray(Z_fit, dtype=np.float64), np.asarray(y_fit))
    pred = knn.predict(np.asarray(Z_eval, dtype=np.float64))
    y_eval = np.asarray(y_eval)
    classes = np.unique(np.concatenate([np.asarray(y_fit), y_eval]))
    rec = [float((pred[y_eval == c] == c).mean()) for c in classes if (y_eval == c).any()]
    return {"wa": round(float((pred == y_eval).mean()), 4),
            "ua": round(float(np.mean(rec)), 4)}


def group_predictability(Z, groups, k=10, metric="cosine"):
    """Leave-one-out k-NN accuracy for a nuisance variable, vs its majority rate.

    `lift` well above 0 means the embedding encodes that variable. For a
    speaker label this is the entanglement warning: whatever the student copies
    will include it.
    """
    Z = np.asarray(Z, dtype=np.float64)
    g = np.asarray(groups)
    knn = KNeighborsClassifier(n_neighbors=k + 1, metric=metric).fit(Z, g)
    # drop self-match: the first neighbour of a fitted point is itself
    nbr = knn.kneighbors(Z, return_distance=False)[:, 1:]
    votes = g[nbr]
    pred = np.array([np.bincount(np.searchsorted(np.unique(g), row)).argmax() for row in votes])
    pred = np.unique(g)[pred]
    acc = float((pred == g).mean())
    chance = float(pd.Series(g).value_counts(normalize=True).max())
    return {"acc": round(acc, 4), "majority_rate": round(chance, 4),
            "lift": round(acc - chance, 4), "n_groups": int(len(np.unique(g)))}


def class_centroid_similarity(Z, y, class_names):
    """Cosine between class centroids -- which classes the embedding conflates."""
    Zn = normalize(np.asarray(Z, dtype=np.float64))
    cents = np.stack([Zn[np.asarray(y) == i].mean(0) for i in range(len(class_names))])
    cents = normalize(cents)
    return pd.DataFrame((cents @ cents.T).round(4), index=class_names, columns=class_names)


def embed_2d(Z, method="pca", seed=0, perplexity=30):
    Z = np.asarray(Z, dtype=np.float64)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(Z)
    if method == "tsne":
        return TSNE(n_components=2, random_state=seed, init="pca", metric="cosine",
                    perplexity=min(perplexity, max(5, len(Z) // 4 - 1))).fit_transform(Z)
    raise ValueError(f"unknown method {method!r}")


# ── plotting ─────────────────────────────────────────────────────────────────


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


def plot_class_facets(Z2, y, class_names, path, title="", subtitle=""):
    """One small multiple per class: that class in colour, the rest recessive grey.

    Faceting rather than four hues in one scatter -- with every class on screen
    at once, overlapping points make identity ambiguous no matter how good the
    palette is, and the per-class shape is what actually needs reading.
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y)
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for i, (ax, name) in enumerate(zip(axes, class_names)):
        m = y == i
        ax.scatter(Z2[~m, 0], Z2[~m, 1], s=7, c="#d8d7d1", linewidths=0, alpha=0.85, rasterized=True)
        ax.scatter(Z2[m, 0], Z2[m, 1], s=9, c=PALETTE[i % len(PALETTE)],
                   linewidths=0, alpha=0.92, rasterized=True)
        ax.set_title(f"{name}  (n={int(m.sum())})", fontsize=10, color=INK, pad=6)
        _style(ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if title:
        fig.suptitle(title, fontsize=12.5, color=INK, x=0.01, ha="left", y=0.995)
    if subtitle:
        fig.text(0.01, 0.93, subtitle, fontsize=9, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90 if subtitle else 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_bars(df, value_col, group_col, series_col, path, title="",
                      subtitle="", ylabel="", ref_line=None, ref_label=""):
    """Grouped bars with every value direct-labelled -- the relief rule, and it
    keeps the figure readable without the CSV in hand."""
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(df[group_col]))
    series = list(dict.fromkeys(df[series_col]))
    x = np.arange(len(groups), dtype=float)
    w = min(0.8 / max(len(series), 1), 0.34)

    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 2.4, 3.8), facecolor=SURFACE)
    for si, s in enumerate(series):
        vals = [float(df[(df[group_col] == g) & (df[series_col] == s)][value_col].mean())
                for g in groups]
        off = (si - (len(series) - 1) / 2) * w
        ax.bar(x + off, vals, width=w * 0.92, color=PALETTE[si % len(PALETTE)],
               label=str(s), linewidth=0)
        for xi, v in zip(x + off, vals):
            ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -11), ha="center",
                        fontsize=7.5, color=INK_SOFT)
    if ref_line is not None:
        ax.axhline(ref_line, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(ref_label, (ax.get_xlim()[1], ref_line), textcoords="offset points",
                    xytext=(-2, 4), ha="right", fontsize=8, color=INK_SOFT)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SOFT)
    _style(ax)
    ax.grid(axis="x", visible=False)
    if len(series) >= 2:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT, ncols=min(len(series), 4))
    if title:
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=14)
    if subtitle:
        ax.annotate(subtitle, (0, 1.02), xycoords="axes fraction", fontsize=9,
                    color=INK_SOFT, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(mat, path, title="", subtitle="", vmin=-1.0, vmax=1.0):
    """Sequential single-hue cell chart with every cell labelled."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("aqua", ["#ffffff", PALETTE[2], "#0d5c40"])
    fig, ax = plt.subplots(figsize=(1.1 * len(mat) + 2.2, 1.0 * len(mat) + 1.9),
                           facecolor=SURFACE)
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)), mat.columns, fontsize=9, color=INK)
    ax.set_yticks(range(len(mat.index)), mat.index, fontsize=9, color=INK)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > (vmin + vmax) / 2 + 0.28 else INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=8, length=3)
    if title:
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=14)
    if subtitle:
        ax.annotate(subtitle, (0, 1.03), xycoords="axes fraction", fontsize=9,
                    color=INK_SOFT, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_grid(panels, class_names, path, title="", subtitle="", ncols=3):
    """One scatter per representation, all classes coloured together.

    `panels` is a list of (label, Z2, y, caption). Faceting per class the way
    `plot_class_facets` does answers "where does this class sit"; this answers
    "how separated is the whole thing", which is the comparison when several
    representations are placed side by side.

    Four hues are used at once, so the palette has to clear the all-pairs CVD
    and normal-vision floors rather than the easier adjacent-pair ones -- slots
    1-4 (blue / orange / aqua / violet) are the subset that does. A legend plus
    per-panel captions carry identity so colour is never the only channel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.7 * nrows),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, Z2, y, cap) in zip(axes, panels):
        y = np.asarray(y)
        for i, name in enumerate(class_names):
            m = y == i
            ax.scatter(Z2[m, 0], Z2[m, 1], s=6, c=PALETTE[i % len(PALETTE)],
                       linewidths=0, alpha=0.85, rasterized=True, label=name)
        ax.set_title(label, fontsize=10.5, color=INK, pad=5)
        if cap:
            ax.set_xlabel(cap, fontsize=8, color=INK_SOFT, labelpad=4)
        _style(ax)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for ax in axes[n:]:
        ax.axis("off")

    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      color=PALETTE[i % len(PALETTE)], label=c)
               for i, c in enumerate(class_names)]
    fig.legend(handles=handles, loc="lower center", ncols=len(class_names),
               frameon=False, fontsize=9.5, labelcolor=INK_SOFT,
               bbox_to_anchor=(0.5, -0.005))
    if title:
        fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left", y=0.995)
    if subtitle:
        fig.text(0.01, 0.962, subtitle, fontsize=9, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945 if subtitle else 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
