"""
Embedding analysis helpers, used for KD debugging.

The questions here aren't about accuracy. They are: does this embedding
separate the classes or mostly the speakers, does the class structure survive
unseen speakers, and does the student end up shaped like the teacher.

Takes float matrices and int label arrays, so nothing is IEMOCAP specific.
Every plot also dumps its numbers to csv so a figure is never the only record.

Two metrics that need a word:

group_predictability - leave-one-out kNN accuracy for a nuisance variable
(speaker, session), compared to the majority rate. If a teacher embedding
predicts speaker at 90% it is handing the student structure that can't transfer
to held-out speakers, no matter how good the class accuracy is.

knn_transfer - fit on one split, score another, no training. Tells you how much
class structure is actually portable, separate from what a trained head could
dig out.

Colours are slots 1-4 of the categorical palette. Aqua is below 3:1 on the
light background so every figure gets a legend and direct labels too.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# categorical palette, light mode
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c2367f", "#a8760a"]
# first four are the class palette, last two extend it for the six-panel figures
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e0"


def cosine_separability(Z, y):
    """mean cosine within vs between classes, rows L2-normalised.

    `gap` is the number to look at. Scale-free, so it compares across
    embeddings of different width.
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
    """silhouette in [-1,1]. subsampled above max_n, it's O(n^2)."""
    Z, y = np.asarray(Z, dtype=np.float64), np.asarray(y)
    if len(y) > max_n:
        idx = np.random.RandomState(seed).choice(len(y), max_n, replace=False)
        Z, y = Z[idx], y[idx]
    return round(float(silhouette_score(Z, y, metric=metric)), 4)


def knn_transfer(Z_fit, y_fit, Z_eval, y_eval, k=10, metric="cosine"):
    """fit kNN on one split, score another. no training involved."""
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(np.asarray(Z_fit, dtype=np.float64), np.asarray(y_fit))
    pred = knn.predict(np.asarray(Z_eval, dtype=np.float64))
    y_eval = np.asarray(y_eval)
    classes = np.unique(np.concatenate([np.asarray(y_fit), y_eval]))
    rec = [float((pred[y_eval == c] == c).mean()) for c in classes if (y_eval == c).any()]
    return {"wa": round(float((pred == y_eval).mean()), 4),
            "ua": round(float(np.mean(rec)), 4)}


def group_predictability(Z, groups, k=10, metric="cosine"):
    """leave-one-out kNN accuracy for a nuisance variable, against its majority
    rate. lift well above 0 means the embedding encodes it. For speaker labels
    that's the warning sign - the student will copy it too.
    """
    Z = np.asarray(Z, dtype=np.float64)
    g = np.asarray(groups)
    knn = KNeighborsClassifier(n_neighbors=k + 1, metric=metric).fit(Z, g)
    # drop self match, first neighbour of a fitted point is itself
    nbr = knn.kneighbors(Z, return_distance=False)[:, 1:]
    votes = g[nbr]
    pred = np.array([np.bincount(np.searchsorted(np.unique(g), row)).argmax() for row in votes])
    pred = np.unique(g)[pred]
    acc = float((pred == g).mean())
    chance = float(pd.Series(g).value_counts(normalize=True).max())
    return {"acc": round(acc, 4), "majority_rate": round(chance, 4),
            "lift": round(acc - chance, 4), "n_groups": int(len(np.unique(g)))}


def class_centroid_similarity(Z, y, class_names):
    """cosine between class centroids, i.e. which classes get conflated."""
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
    """one panel per class, that class coloured and the rest grey.

    Faceted instead of four hues in one scatter, because with everything on
    screen at once the overlaps make it impossible to tell which point is which.
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
    """grouped bars, every value labelled so the figure is readable on its own."""
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
        # legend below the axes, inside it sits on top of the bars
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT,
                  ncols=min(len(series), 3), loc="upper center",
                  bbox_to_anchor=(0.5, -0.09), borderaxespad=0.0)
    if title:
        ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=14)
    if subtitle:
        ax.annotate(subtitle, (0, 1.02), xycoords="axes fraction", fontsize=9,
                    color=INK_SOFT, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_heatmap(mat, path, title="", subtitle="", vmin=-1.0, vmax=1.0):
    """single-hue heatmap, every cell labelled."""
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
    """one scatter per representation, all classes coloured together.

    panels is a list of (label, Z2, y, caption). plot_class_facets answers
    "where does this class sit"; this one answers "how separated is the whole
    thing", which is what you want when comparing several embeddings.

    Four hues at once, so there's a legend and per-panel captions as well -
    colour shouldn't be the only channel.
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
