"""
Do train, val and test sit in the same region of the teacher feature space?

The student is asked to reproduce, on training utterances, a geometry the
teacher produced on those same utterances. That only pays off on test if the
teacher representation puts the three splits in the same place. Under the
speaker-independent protocol there is reason to doubt it - the splits share no
speakers at all and their class priors differ (KL(val, test) = 0.037, four
times KL(train, test)).

Six representations get pooled across all three splits, embedded together so
they compare directly, and coloured by split rather than by class. The question
isn't whether the classes separate, it's whether this is one distribution.

The headline number is split predictability: leave-one-out kNN accuracy at
guessing which split a sample came from, against the majority rate. Chance
means the splits are interchangeable. Well above chance means the feature space
encodes which session a recording came from, and whatever the student copies
from train carries that along.

Per-class centroid cosine between splits is the sharper version: even if the
clouds overlap, does angry point the same way in train as in test? That's what
a feature-KD target actually has to get right.

One trap, and the reason centroid_norm_* is recorded. The 2048-d features are
standardised on train stats, so the global train centroid is pinned at the
origin by construction - its mean unit vector has norm 0.01 to 0.03 against
0.42 to 0.49 at 64-d. A direction that short is noise, so the global cos_train_*
columns mean nothing when centroid_norm_train is near zero, and only the 64-d
rows and the per-class cosines carry signal there.

The teacher rows are deterministic, the student rows are not. cuDNN alone moved
per-class neutral drift from 0.112 to 0.289 between two identical runs, which
is bigger than the CE vs KD difference it was being used to argue about. So
students are trained over several seeds and reported as mean +- sd, with every
seed kept in shift_seeds.csv. Any student column whose sd is about the size of
the CE vs KD gap should be read as no difference measured.

Outputs (outputs/iemocap/analysis/):
    fig_split_shift.png          six panels, pooled embedding coloured by split
    fig_centroid_drift.png       per-class train-vs-test centroid cosine
    shift_metrics.csv            split predictability, centroid cosines, mean +- sd
    shift_per_class.csv          per-class centroid cosine, every split pair
    shift_seeds.csv              one row per (representation, seed), unaggregated

Usage:
    python src/iemocap/analysis/distribution_shift.py
    python src/iemocap/analysis/distribution_shift.py --seeds 42 43 44 45 46
    python src/iemocap/analysis/distribution_shift.py --no-embed      # metrics only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.repr_analysis import (  # noqa: E402
    embed_2d, group_predictability, plot_embedding_grid, plot_grouped_bars,
)
from iemocap.analysis.cluster_features import teacher_reps, train_student  # noqa: E402
from iemocap.paths import IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    CLASSES, EPOCHS, load_inputs, load_teacher_signals, normalizer,
)

OUT = IEMOCAP_OUTPUTS / "analysis"
SPLITS = ("train", "val", "test")
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


def centroid_cos(a, b):
    """Cosine between the mean directions of two point clouds."""
    ca, cb = normalize(a).mean(0), normalize(b).mean(0)
    return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))


def rep_metrics(per_split, k):
    """Split predictability, split-pair centroid cosines and per-class drift
    for one concrete embedding of the dataset."""
    Z = np.concatenate([per_split[s][0] for s in SPLITS])
    split_id = np.concatenate([np.full(len(per_split[s][1]), i)
                               for i, s in enumerate(SPLITS)])
    gp = group_predictability(Z, split_id, k=k)

    row = {"dim": Z.shape[1], "split_knn_acc": gp["acc"],
           "majority": gp["majority_rate"], "split_lift": gp["lift"]}
    for a, b in PAIRS:
        row[f"cos_{a}_{b}"] = centroid_cos(per_split[a][0], per_split[b][0])
    # how long the mean unit vector is; near zero means the cos_* above is noise
    for sp in SPLITS:
        row[f"centroid_norm_{sp}"] = float(np.linalg.norm(normalize(per_split[sp][0]).mean(0)))

    per_class = {}
    for ci, cname in enumerate(CLASSES):
        per_class[cname] = {
            f"{a}_{b}": centroid_cos(per_split[a][0][per_split[a][1] == ci],
                                     per_split[b][0][per_split[b][1] == ci])
            for a, b in PAIRS}
    return row, per_class


def agg(dicts, index):
    """mean and sd across seeds, flattened into one row."""
    out = dict(index)
    out["n_seeds"] = len(dicts)
    for key in dicts[0]:
        vals = np.array([d[key] for d in dicts], dtype=float)
        out[key] = round(float(vals.mean()), 4)
        if key not in ("dim", "majority"):
            out[f"{key}_sd"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
    return out


def main():
    ap = argparse.ArgumentParser(description="Split-level distribution shift in the "
                                             "teacher's and student's feature spaces.")
    ap.add_argument("--embed", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--no-embed", action="store_true", help="metrics only, skip the figures")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="student seeds; teacher rows are deterministic")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # name -> list of instances (one per seed; teachers have exactly one)
    reps = {}
    for key in ("audio_mean_l27", "last_token"):
        print(f"teacher {key} ...", flush=True)
        hi, lo = teacher_reps(key)
        reps[f"teacher {key} [2048]"] = [hi]
        reps[f"teacher {key} [64]"] = [lo]

    Xtr, ytr, ids_tr = load_inputs("train")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    teach = load_teacher_signals(ids_tr)
    data = (Xtr, ytr, mu, sd, teach["z_audio"], evalsets)
    for kind, name in (("ce", "student CE [64]"), ("featkd", "student Feature-KD [64]")):
        reps[name] = []
        for seed in args.seeds:
            print(f"training {name} seed {seed} ...", flush=True)
            reps[name].append(train_student(kind, data, args.epochs, seed=seed))

    rows, per_class, seed_rows, panels = [], [], [], []
    for name, instances in reps.items():
        seeds = args.seeds if len(instances) > 1 else [0]
        mets = [rep_metrics(inst, args.k) for inst in instances]

        rows.append(agg([m[0] for m in mets], {"representation": name}))
        for cname in CLASSES:
            per_class.append(agg([m[1][cname] for m in mets],
                                 {"representation": name, "class": cname}))
        for seed, (r, pc) in zip(seeds, mets):
            seed_rows.append({"representation": name, "seed": seed,
                              **{k: round(v, 4) for k, v in r.items()},
                              **{f"{c}_train_test": round(pc[c]["train_test"], 4)
                                 for c in CLASSES}})

        if not args.no_embed:
            Z = np.concatenate([instances[0][sp][0] for sp in SPLITS])
            split_id = np.concatenate([np.full(len(instances[0][sp][1]), i)
                                       for i, sp in enumerate(SPLITS)])
            r = rows[-1]
            note = (f"split k-NN {r['split_knn_acc']:.3f} vs {r['majority']:.3f} chance"
                    f"   (lift {r['split_lift']:+.3f}"
                    + (f" +- {r['split_lift_sd']:.3f}" if len(instances) > 1 else "") + ")")
            print(f"  embedding {name} ({len(Z)} points) ...", flush=True)
            panels.append((name, embed_2d(Z, method=args.embed, seed=0), split_id, note))

    m = pd.DataFrame(rows)
    pc = pd.DataFrame(per_class)
    m.to_csv(OUT / "shift_metrics.csv", index=False)
    pc.to_csv(OUT / "shift_per_class.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "shift_seeds.csv", index=False)

    if not args.no_embed:
        plot_embedding_grid(panels, list(SPLITS), OUT / "fig_split_shift.png",
                            title="Do the three splits occupy the same space?",
                            subtitle="All 5,531 utterances embedded together per representation, "
                                     "coloured by split. Speaker-independent protocol. "
                                     "Students shown at the first seed.")
        bars = pc.melt(id_vars=["representation", "class"], value_vars=["train_test"],
                       var_name="pair", value_name="cos")
        plot_grouped_bars(bars, "cos", "class", "representation",
                          OUT / "fig_centroid_drift.png",
                          title="Does each class sit in the same direction in train and test?",
                          subtitle=f"Cosine between per-class centroids, train vs test. 1.0 = no "
                                   f"drift. Students averaged over {len(args.seeds)} seeds.",
                          ylabel="train-test centroid cosine", ref_line=1.0, ref_label="no drift")

    print("\n=== split predictability (can k-NN tell which split a sample is from?) ===")
    print(m[["representation", "dim", "n_seeds", "split_knn_acc", "majority", "split_lift",
             "split_lift_sd", "cos_train_val", "cos_train_test", "cos_val_test",
             "centroid_norm_train"]].to_string(index=False))
    print("  (cos_* is noise wherever centroid_norm_train is near zero -- see docstring)")
    print("\n=== per-class centroid cosine, train vs test (mean +- sd over seeds) ===")
    show = pc.copy()
    show["v"] = [f"{a:.3f} +-{b:.3f}" for a, b in zip(show.train_test, show.train_test_sd)]
    print(show.pivot(index="representation", columns="class", values="v")
              .reindex(columns=CLASSES).to_string())
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
