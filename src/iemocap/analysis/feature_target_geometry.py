"""
What does the 64-d Feature-KD target actually look like, and would MSE differ
from the cosine loss we use?

The two losses are closer than they appear. For L2-normalised vectors,

    || a/|a| - b/|b| ||^2  =  2 - 2 cos(a, b)

so "MSE on normalised features" IS the cosine loss up to a factor of two. The
only thing raw MSE adds is a demand that the student reproduce the target's
MAGNITUDE as well as its direction. Whether that is signal or noise is an
empirical question about the target, and it is answered here without training
anything:

    energy split      how the squared-error budget MSE would optimise divides
                      into DC (the direction every sample shares), between-class
                      (the part that carries the label) and within-class (the
                      part that does not).
    norm information  whether |z| alone predicts the class at all, and whether
                      it merely tracks the teacher's confidence.

If between-class energy is a small slice and the norm carries no class signal,
raw MSE spends most of the student's 96k parameters reproducing quantities that
cannot improve its decision boundary -- and the student's bottleneck is a
BatchNorm output, whose scale is fixed by construction, so it cannot even
comply without distorting the part that matters.

Teacher z comes from Linear -> LayerNorm -> GELU, so it is per-sample
normalised and mostly non-negative before anything here touches it. Student z
is BatchNorm1d, so it is per-dimension zero-mean and symmetric. Those are
different geometries, which is the second reason raw MSE is a strange fit.

Outputs (outputs/iemocap/analysis/):
    target_geometry.csv        energy split and norm statistics per target
    target_norm_by_class.csv   per-class mean |z|, and what a norm-only classifier gets

Usage:
    python src/iemocap/analysis/feature_target_geometry.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.analysis.cluster_features import teacher_reps  # noqa: E402
from iemocap.paths import IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import CLASSES  # noqa: E402

OUT = IEMOCAP_OUTPUTS / "analysis"
KEYS = ("audio_mean_l27", "last_token")


def energy_split(Z, y):
    """Decompose E||z||^2 into DC + between-class + within-class."""
    mu = Z.mean(0)
    total = float((Z ** 2).sum(1).mean())
    dc = float((mu ** 2).sum())
    Zc = Z - mu
    between = 0.0
    for c in np.unique(y):
        m = Zc[y == c].mean(0)
        between += (y == c).mean() * float((m ** 2).sum())
    within = float((Zc ** 2).sum(1).mean()) - between
    return {"energy_total": round(total, 4),
            "frac_dc": round(dc / total, 4),
            "frac_between": round(between / total, 4),
            "frac_within": round(within / total, 4)}


def norm_stats(Z, y, k=10):
    n = np.linalg.norm(Z, axis=1)
    # can the class be read off the magnitude alone? 1-D leave-one-out k-NN.
    order = np.argsort(n)
    ns, ys = n[order], y[order]
    hits = np.zeros(len(n), dtype=bool)
    for i in range(len(ns)):
        lo, hi = max(0, i - k), min(len(ns), i + k + 1)
        nb = np.concatenate([ys[lo:i], ys[i + 1:hi]])
        hits[i] = np.bincount(nb, minlength=len(CLASSES)).argmax() == ys[i]
    ua = float(np.mean([hits[ys == c].mean() for c in range(len(CLASSES))]))
    return {"norm_mean": round(float(n.mean()), 3),
            "norm_sd": round(float(n.std()), 3),
            "norm_cv": round(float(n.std() / n.mean()), 4),
            "frac_negative_dims": round(float((Z < 0).mean()), 4),
            "norm_only_knn_ua": round(ua, 4)}, n


def main():
    ap = argparse.ArgumentParser(description="Geometry of the 64-d Feature-KD target.")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, per_class = [], []
    for key in KEYS:
        _, lo = teacher_reps(key)
        for split in ("train", "test"):
            Z, y = lo[split]
            st, n = norm_stats(Z, y, args.k)
            rows.append({"target": f"teacher {key} [64]", "split": split, "n": len(y),
                         **energy_split(Z, y), **st})
            if split == "train":
                for ci, c in enumerate(CLASSES):
                    per_class.append({"target": f"teacher {key} [64]", "class": c,
                                      "mean_norm": round(float(n[y == ci].mean()), 3),
                                      "sd_norm": round(float(n[y == ci].std()), 3)})

    df = pd.DataFrame(rows)
    pc = pd.DataFrame(per_class)
    df.to_csv(OUT / "target_geometry.csv", index=False)
    pc.to_csv(OUT / "target_norm_by_class.csv", index=False)

    print("\n=== where MSE's squared-error budget would go ===")
    print(df[["target", "split", "energy_total", "frac_dc", "frac_between",
              "frac_within"]].to_string(index=False))
    print("\n=== is the magnitude informative? (chance UA = 0.25) ===")
    print(df[["target", "split", "norm_mean", "norm_sd", "norm_cv",
              "frac_negative_dims", "norm_only_knn_ua"]].to_string(index=False))
    print("\n=== per-class mean |z| on train ===")
    print(pc.pivot(index="target", columns="class",
                   values="mean_norm").reindex(columns=CLASSES).to_string())
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
