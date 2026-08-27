"""
How small can the target get before the student stops being able to copy it?

B_pca in target_variants.py swapped the label-trained bottleneck for an
unsupervised PCA-64 and gained 1.8 points. PCA-64 still keeps far more than
emotion, though - speaker, channel and phonetic variation all sit in those
components - and the feature loss orders the student to reproduce every one of
them. This keeps only the leading components, chosen by the variance they
retain, and shrinks the student's projection to match, so what the student
chases is closer to the few directions that carry class structure.

Three numbers per variant, which separate the three ways this can fail:

    the target's own k-NN and linear probe, i.e. what a perfect student scores
    r2_train / r2_test, how much of the target the student reproduces
    test_ua through the usual stage-2 head

Capacity shows up as a low r2_train, generalisation as the train-test gap, and
a target not worth copying as a low k-NN with a high r2.

One fold, five seeds, everything else the fixed protocol.

    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/lowdim_target.py
    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/lowdim_target.py --variants v50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.losses import kd_feature_loss  # noqa: E402
from common.models.audio_student import DSResNetSE  # noqa: E402
from iemocap.paths import PROTOCOL, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, LR, N_CLASSES, PROJ_DIM, SMALL_KW,
    WEIGHT_DECAY, available_splits, load_inputs, normalizer,
)
from iemocap.student.target_variants import (  # noqa: E402
    centre, metrics, stage2, teacher_features,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "lowdim_target.csv"
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]

# name -> variance to keep, or a fixed width. pca64 is B_pca again, as control.
VARIANTS = {"d4": ("dim", 4), "d8": ("dim", 8), "d16": ("dim", 16),
            "v50": ("var", 0.50), "v70": ("var", 0.70), "pca64": ("dim", PROJ_DIM)}


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def knn_ua(ztr, ytr, zte, yte, k=10):
    """cosine k-NN fitted on the target's train side, scored on its test side."""
    nn_idx = np.argsort(-(unit(zte) @ unit(ztr).T), axis=1)[:, :k]
    pred = [np.bincount(ytr[r], minlength=N_CLASSES).argmax() for r in nn_idx]
    return round(float(recall_score(yte, pred, average="macro", zero_division=0)), 4)


def probe_ua(ztr, ytr, zte, yte):
    """what a linear readout of the target alone is worth."""
    lr = LogisticRegression(max_iter=2000).fit(ztr, ytr)
    return round(float(recall_score(yte, lr.predict(zte), average="macro",
                                    zero_division=0)), 4)


def r2(zs, zt):
    """variance of the target explained, after the one global scale a cosine
    loss leaves free. 0 is emitting a constant, 1 is exact. Both sides are
    already centred, so the denominator is the target's own variance."""
    a = float((zs * zt).sum() / ((zs * zs).sum() + 1e-12))
    return round(float(1 - ((a * zs - zt) ** 2).sum() / ((zt ** 2).sum() + 1e-12)), 4)


def pca_bank(Xtr):
    """one SVD, sliced per variant."""
    p = PCA(svd_solver="full", random_state=0).fit(Xtr.numpy())
    return p, np.cumsum(p.explained_variance_ratio_)


def pca_target(p, tf, k):
    return {s: torch.from_numpy(
        (tf[s]["h"].numpy() - p.mean_) @ p.components_[:k].T).float() for s in SPLITS}


def run_one(target, dim, data, seed):
    """target_variants.run_one with the projection width following the target."""
    Xtr, ytr, mu, sd, evalsets = data
    t_z = centre(target["train"], "perdim", target["train"].mean(0, keepdim=True)).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=dim,
                       dropout=DROPOUT, **SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = spec_augment(((Xtr[idx].float() - mu) / sd).unsqueeze(1).to(DEVICE))
            z, _ = model(xb)
            opt.zero_grad()
            kd_feature_loss(centre(z, "perdim"), t_z[idx]).backward()
            opt.step()
        sched.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for s, (Xs, ys) in evalsets.items():
            zs = []
            for i in range(0, len(Xs), 256):
                xb = ((Xs[i:i + 256].float() - mu) / sd).unsqueeze(1).to(DEVICE)
                zs.append(model(xb)[0].cpu())
            out[s] = (torch.cat(zs).numpy(), ys.numpy())
    return out, sum(q.numel() for q in model.parameters())


def main():
    ap = argparse.ArgumentParser(description="Shrink the PCA target, keep the rest.")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tf, _ = teacher_features()
    Xtr, ytr, ids = load_inputs("train")
    if list(tf["train"]["ids"]) != list(ids):
        raise RuntimeError("teacher feature ids do not match the log-mel cache ids")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)
    t_logits = tf["train"]["logits"]

    p, cum = pca_bank(tf["train"]["h"])
    print(f"protocol={PROTOCOL}  seeds={args.seeds}\nvariance kept: "
          + "  ".join(f"{k}d {cum[k - 1]:.2f}" for k in (2, 4, 8, 16, 32, 64, 128)),
          flush=True)

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for name in args.variants:
        kind, val = VARIANTS[name]
        k = int(np.searchsorted(cum, val) + 1) if kind == "var" else int(val)
        target = pca_target(p, tf, k)
        t_mu = target["train"].mean(0, keepdim=True)
        ceil = {"knn_ua": knn_ua(target["train"].numpy(), ytr,
                                 target["test"].numpy(), tf["test"]["y"]),
                "probe_ua": probe_ua(target["train"].numpy(), ytr,
                                     target["test"].numpy(), tf["test"]["y"])}
        print(f"{name}: k={k}  var={cum[k - 1]:.3f}  target k-NN {ceil['knn_ua']}  "
              f"linear {ceil['probe_ua']}", flush=True)

        for seed in args.seeds:
            t0 = time.time()
            out, n_par = run_one(target, k, data, seed)
            predict = stage2(out["train"][0], out["train"][1], t_logits, seed)
            r = {"protocol": PROTOCOL, "variant": name, "dim": k,
                 "var_kept": round(float(cum[k - 1]), 4), "params": n_par,
                 "seed": seed, **ceil, "seconds": round(time.time() - t0, 1)}
            for s in SPLITS:
                z, y = out[s]
                r.update(metrics(y, predict(z), s))
                r[f"r2_{s}"] = r2(centre(torch.from_numpy(z), "perdim").numpy(),
                                  centre(target[s], "perdim", t_mu).numpy())
            new.append(r)
            print(f"  seed {seed}: test UA {r['test_ua']:.4f}  "
                  f"r2 {r['r2_train']:.3f}/{r['r2_test']:.3f}  ({r['seconds']:.0f}s)",
                  flush=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(
                RUNS_CSV, index=False)

    df = pd.read_csv(RUNS_CSV)
    print("\n=== test UA, mean over seeds ===")
    print(df.groupby("variant").agg(
        n=("seed", "size"), dim=("dim", "first"), var=("var_kept", "first"),
        knn=("knn_ua", "first"), lin=("probe_ua", "first"),
        test_ua=("test_ua", "mean"), sd=("test_ua", "std"),
        r2_train=("r2_train", "mean"), r2_test=("r2_test", "mean"),
    ).sort_values("dim").round(4).to_string())
    print(f"\n-> {RUNS_CSV}")


if __name__ == "__main__":
    main()
