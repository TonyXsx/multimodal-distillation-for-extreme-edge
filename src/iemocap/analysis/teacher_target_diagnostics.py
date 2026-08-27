"""
What the 64-d KD target is made of, and how much of it reaches the student.

Written for the HuBERT control, but it runs on both arms, because the point is
the comparison: the omni teacher wins every measurement of target quality and
still loses the student comparison, so the target has to be looked at rather
than assumed.

Seven questions, one CSV each, all read-only. Nothing is retrained: teacher
targets come from the saved probes, student embeddings from the z_cache the
fixed protocol already wrote.

    quality      is the qwen target the better target on its own terms
    shift        does the target move between train and test, and does the
                 teacher survive the move
    energy       how much of the target is a shared offset, the class label,
                 and everything else
    gelu         the shared offset is an artefact of reading the probe after
                 its GELU. what pre-GELU, centring and whitening do to it
    simulation   how close a student would have to get before a simple readout
                 is worth having
    ridge        can any audio-only mapping to the target generalise, with the
                 tiny student replaced by a frozen strong front end
    fidelity     how close the students we trained actually got

Test-side teacher targets are recomputed here from the saved probe. They are
measurement only; no student was ever trained against them.

    python src/iemocap/analysis/teacher_target_diagnostics.py
    python src/iemocap/analysis/teacher_target_diagnostics.py --folds 1 2
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.probe import Probe  # noqa: E402
from common.repr_analysis import cosine_separability  # noqa: E402
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_OUTPUTS  # noqa: E402

OUT = IEMOCAP_OUTPUTS / "analysis" / "teacher_compare"
TEACHERS = ("qwen", "hubert")
# the feature each arm distils, at the same relative depth in its own stack
KEY = {"qwen": "audio_mean_l27", "hubert": "hubert_mean_l18"}
# audio-target methods only, the two arms have no last_token in common
METHODS = ("ce", "feature_kd_audio", "full_kd_audio", "feature_only_audio")
RNG = np.random.default_rng(0)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def ua(y, p):
    return round(float(recall_score(y, p, average="macro")), 4)


def collapse(z):
    """what a student scores on cosine by emitting one constant vector."""
    zn = unit(z)
    c = zn.mean(0)
    return float((zn @ (c / np.linalg.norm(c))).mean())


def fit_pct(z_hat, z_true, base):
    cos = float((unit(z_hat) * unit(z_true)).sum(1).mean())
    return cos, (cos - base) / (1 - base) * 100


def knn_ua(ztr, ytr, zte, yte, k=10):
    return ua(yte, KNeighborsClassifier(k, metric="cosine").fit(ztr, ytr).predict(zte))


def paths(teacher, fold):
    suf = "" if teacher == "qwen" else "_hubert"
    return (glob.glob(str(IEMOCAP_DATA / f"teacher_features{suf}_loso{fold}" / "*LORA*"))[0],
            IEMOCAP_DATA / f"teacher_probe{suf}_loso{fold}" / "bottleneck" / "adapted" / KEY[teacher],
            IEMOCAP_DATA / f"student_loso{fold}" / f"z_cache{suf}")


def load(teacher, fold):
    """the 2048/1024-d feature, the 64-d target, and the two activations that
    bracket it, for both splits."""
    feat, pdir, _ = paths(teacher, fold)
    ck = torch.load(pdir / "checkpoint.pt", weights_only=False, map_location="cpu")
    m = Probe(ck["in_dim"], [ck["bottleneck"]], len(ck["classes"]), dropout=0.1).eval()
    m.load_state_dict(ck["state_dict"])
    lin, ln = m.blocks[0][0], m.blocks[0][1]

    out = {}
    for s in ("train", "test"):
        d = torch.load(f"{feat}/{s}_features.pt", weights_only=False, map_location="cpu")
        h = d["features"][KEY[teacher]].float()
        with torch.no_grad():
            pre = ln(lin((h - ck["mu"]) / ck["sd"]))       # post-LayerNorm, pre-GELU
            z = F.gelu(pre)                                 # what is distilled
            logits = m.head(z)
        out[s] = {"h": h.numpy(), "z": z.numpy(), "pre": pre.numpy(),
                  "y": d["labels"].numpy(), "pred": logits.argmax(1).numpy(),
                  "logits": d["features"]["logits"].float()}
    return out


# 1. quality
def quality(t, fold, teacher):
    ztr, ytr, zte, yte = t["train"]["z"], t["train"]["y"], t["test"]["z"], t["test"]["y"]
    lg, y = t["train"]["logits"], torch.from_numpy(ytr)
    p2 = F.softmax(lg / 2.0, 1)
    return {"fold": fold, "teacher": teacher,
            "probe_ua_train": ua(ytr, t["train"]["pred"]),
            "probe_ua_test": ua(yte, t["test"]["pred"]),
            "gap_train": cosine_separability(ztr, ytr)["gap"],
            "gap_test": cosine_separability(zte, yte)["gap"],
            "knn10_test_ua": knn_ua(ztr, ytr, zte, yte),
            "collapse_train": round(collapse(ztr), 4),
            "collapse_test": round(collapse(zte), 4),
            "logit_train_acc": round(float((lg.argmax(1) == y).float().mean()), 4),
            "logit_pmax_T2": round(float(p2.max(1).values.mean()), 4),
            "logit_nontarget_mass_T2": round(
                float((1 - p2.gather(1, y[:, None]).squeeze()).mean()), 4)}


# 2. shift
def shift(t, fold, teacher):
    """the marginal distribution barely moves; what moves is how tight the
    classes are. a domain classifier says how far apart the splits really are."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    ztr, ytr, zte, yte = t["train"]["z"], t["train"]["y"], t["test"]["z"], t["test"]["y"]
    mu, sd = ztr.mean(0), ztr.std(0) + 1e-9
    d = np.abs(zte.mean(0) - mu) / sd

    n = min(len(ztr), len(zte))
    idx = RNG.choice(len(ztr), n, replace=False)
    X = StandardScaler().fit_transform(np.vstack([ztr[idx], zte[:n]]))
    dom = np.r_[np.zeros(n), np.ones(n)]
    prob = cross_val_predict(LogisticRegression(max_iter=2000), X, dom, cv=5,
                             method="predict_proba")[:, 1]

    cs = [float(unit(ztr[ytr == c].mean(0)) @ unit(zte[yte == c].mean(0)))
          for c in np.unique(ytr)]
    return {"fold": fold, "teacher": teacher,
            "dim_shift_mean_sd": round(float(d.mean()), 4),
            "dim_shift_max_sd": round(float(d.max()), 4),
            "sd_ratio": round(float((zte.std(0) / sd).mean()), 4),
            "norm_ratio": round(float(np.linalg.norm(zte, axis=1).mean()
                                      / np.linalg.norm(ztr, axis=1).mean()), 4),
            "split_auc": round(float(roc_auc_score(dom, prob)), 4),
            "centroid_cos_mean": round(float(np.mean(cs)), 4),
            "centroid_cos_min": round(float(np.min(cs)), 4),
            "global_dir_cos": round(float(unit(unit(ztr).mean(0)) @ unit(unit(zte).mean(0))), 4),
            "gap_train": cosine_separability(ztr, ytr)["gap"],
            "gap_test": cosine_separability(zte, yte)["gap"],
            "knn10_test_ua": knn_ua(ztr, ytr, zte, yte)}


# 3. energy
def energy(z, y):
    tot = float((z ** 2).sum(1).mean())
    mu = z.mean(0)
    zc = z - mu
    var = float((zc ** 2).sum(1).mean())
    cm = np.stack([zc[y == c].mean(0) for c in np.unique(y)])
    w = np.array([(y == c).mean() for c in np.unique(y)])
    between = float((w[:, None] * cm ** 2).sum())
    ev = np.linalg.svd(zc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return {"shared_pct": round(100 * float(mu @ mu) / tot, 2),
            "between_pct_of_centred": round(100 * between / var, 2),
            "within_pct_of_centred": round(100 * (var - between) / var, 2),
            "pc1_pct": round(100 * float(ev[0]), 2),
            "pc5_pct": round(100 * float(ev[:5].sum()), 2),
            "eff_dim_95pct": int((np.cumsum(ev) < 0.95).sum() + 1)}


# 4. gelu
def variants(t):
    """four ways to hand the same probe activation to the student."""
    ztr, zte = t["train"]["z"], t["test"]["z"]
    mu = ztr.mean(0)
    # whitening is fitted on train only, like every other statistic here
    zc = ztr - mu
    U, S, Vt = np.linalg.svd(zc, full_matrices=False)
    W = Vt.T @ np.diag(1.0 / (S / np.sqrt(len(zc)) + 1e-6)) @ Vt
    return {"post_gelu": (ztr, zte),
            "pre_gelu": (t["train"]["pre"], t["test"]["pre"]),
            "post_gelu_centred": (ztr - mu, zte - mu),
            "post_gelu_whitened": ((ztr - mu) @ W, (zte - mu) @ W)}


# 5. simulate
def degrade(z, mode, a):
    """a=1 exact, a=0 fully degraded. the two ways a student can be wrong."""
    zn = unit(z)
    if mode == "noise":
        return unit(a * zn + (1 - a) * unit(RNG.standard_normal(z.shape)))
    return unit(a * zn + (1 - a) * unit(zn.mean(0, keepdims=True)))


def simulate(t, fold, teacher):
    ztr, ytr, zte, yte = t["train"]["z"], t["train"]["y"], t["test"]["z"], t["test"]["y"]
    base = collapse(zte)
    knn = KNeighborsClassifier(10, metric="cosine").fit(ztr, ytr)
    rows = []
    for mode in ("noise", "collapse"):
        for target in range(10, 101, 10):
            lo, hi = 0.0, 1.0
            for _ in range(40):                       # bisect for the target fit
                a = (lo + hi) / 2
                _, f = fit_pct(degrade(zte, mode, a), zte, base)
                lo, hi = (a, hi) if f < target else (lo, a)
            w = degrade(zte, mode, (lo + hi) / 2)
            cos, f = fit_pct(w, zte, base)
            rows.append({"fold": fold, "teacher": teacher, "mode": mode,
                         "fit_target_pct": target, "fit_actual_pct": round(f, 1),
                         "cos": round(cos, 4), "knn10_ua": ua(yte, knn.predict(w)),
                         "gap": cosine_separability(w, yte)["gap"]})
    return rows


# 6. ridge
def ridge(t_target, t_src, fold, teacher, front_end):
    """frozen front end + linear head, fitted on train only. the tiny student
    with its two limits removed, so what is left is the mapping itself."""
    sc = StandardScaler().fit(t_src["train"]["h"])
    xtr, xte = sc.transform(t_src["train"]["h"]), sc.transform(t_src["test"]["h"])
    ztr, zte = t_target["train"]["z"], t_target["test"]["z"]
    btr, bte = collapse(ztr), collapse(zte)

    best = None                                        # alpha on train, never test
    for alpha in (1e0, 1e1, 1e2, 1e3, 1e4):
        r = Ridge(alpha=alpha).fit(xtr, ztr)
        f = fit_pct(r.predict(xtr), ztr, btr)[1]
        if best is None or f > best[0]:
            best = (f, r, alpha)
    f_tr, r, alpha = best
    p_tr, p_te = r.predict(xtr), r.predict(xte)
    return {"fold": fold, "teacher": teacher, "front_end": front_end, "alpha": alpha,
            "fit_train_pct": round(f_tr, 1),
            "fit_test_pct": round(fit_pct(p_te, zte, bte)[1], 1),
            "gap_test": cosine_separability(p_te, t_target["test"]["y"])["gap"],
            "knn10_test_ua": knn_ua(p_tr, t_target["train"]["y"], p_te, t_target["test"]["y"])}


# -------------------------------------------------------------- 7. fidelity
def fidelity(t, fold, teacher):
    _, _, zc = paths(teacher, fold)
    btr, bte = collapse(t["train"]["z"]), collapse(t["test"]["z"])
    rows = []
    for method in METHODS:
        for f in sorted(glob.glob(str(zc / f"{method}_seed*.pt"))):
            d = torch.load(f, weights_only=False, map_location="cpu")
            r = {"fold": fold, "teacher": teacher, "method": method,
                 "seed": int(Path(f).stem.split("_seed")[1].split("_")[0])}
            for s, base in (("train", btr), ("test", bte)):
                cos, pct = fit_pct(d[s]["z"], t[s]["z"], base)
                r[f"cos_{s}"] = round(cos, 4)
                r[f"fit_{s}_pct"] = round(pct, 1)
            r["gap_test"] = cosine_separability(d["test"]["z"], d["test"]["y"])["gap"]
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="What the 64-d KD target is made of.")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    qual, sh, en, gv, sim, rg, fid = [], [], [], [], [], [], []
    for k in args.folds:
        T = {tc: load(tc, k) for tc in TEACHERS}
        for tc in TEACHERS:
            t = T[tc]
            qual.append(quality(t, k, tc))
            sh.append(shift(t, k, tc))
            for s in ("train", "test"):
                en.append({"fold": k, "teacher": tc, "split": s,
                           **energy(t[s]["z"], t[s]["y"])})
            for name, (a, b) in variants(t).items():
                for s, z in (("train", a), ("test", b)):
                    gv.append({"fold": k, "teacher": tc, "target": name, "split": s,
                               **energy(z, t[s]["y"]),
                               "collapse": round(collapse(z), 4),
                               "gap": cosine_separability(z, t[s]["y"])["gap"]})
                gv[-1]["knn10_test_ua"] = knn_ua(a, t["train"]["y"], b, t["test"]["y"])
            sim += simulate(t, k, tc)
            fid += fidelity(t, k, tc)
            other = "hubert" if tc == "qwen" else "qwen"
            rg.append(ridge(t, T[tc], k, tc, f"{tc} own"))
            rg.append(ridge(t, T[other], k, tc, other))
        print(f"fold {k} done", flush=True)

    for name, rows in (("quality", qual), ("train_test_shift", sh), ("energy", en),
                       ("gelu_variants", gv), ("fidelity_simulation", sim),
                       ("learnability_ridge", rg), ("student_fidelity", fid)):
        path = OUT / f"loso_target_{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"-> {path}")

    q = pd.DataFrame(qual).groupby("teacher").mean(numeric_only=True).round(4)
    print("\n=== target quality, mean over folds ===")
    print(q.drop(columns="fold").to_string())
    print("\n=== student fidelity, mean over folds and seeds ===")
    print(pd.DataFrame(fid).groupby(["teacher", "method"])[
        ["fit_train_pct", "fit_test_pct"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
