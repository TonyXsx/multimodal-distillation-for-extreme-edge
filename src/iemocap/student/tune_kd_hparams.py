"""
KD hyperparameter sweep on IEMOCAP. Is the null result just a tuning failure?

Every KD number so far used T=8 and lambda 1.0/1.0, carried over from FSC
without re-tuning. The FSC write-up warns against exactly that - a negative
claim is only valid at the best hyperparameters, and on FSC moving T from 2 to
8 flipped the conclusion. The two tasks aren't comparable on this axis anyway:
FSC has 31 classes (uniform mass 0.032), IEMOCAP has 4 (uniform 0.25), so T=8
flattens this teacher to a correct-class probability of 0.397 against a 0.25
floor.

Greedy, same order as fsc/student/tune_kd_hparams.py:

    stage 1   logit_kd    T over {1, 2, 4, 8, 16}, lambda_logit = 1
    stage 2   logit_kd    lambda_logit over {0.25, 0.5, 2, 4} at the best T
    stage 3   feature_kd  lambda_feature over {0.25, 0.5, 1, 2, 4}
    stage 4   full_kd     the two winners together

How to read the output, which matters more than the winner. Selection is on val
UA. Test is recorded at every point but never used to choose anything, it's
there so "does the val-optimal setting transfer" can be checked afterwards
instead of assumed.

With one seed and a noise floor of about +/-0.6 points run to run (+/-2 across
seeds), the argmax of this grid isn't trustworthy by itself. The useful thing
is the shape of the surface. A smooth trend in T or lambda means there is a
real effect and the current setting is just off the peak. Scatter with no
structure means the differences are noise and no amount of re-tuning will save
the comparison.

So the question is not which T wins, it's whether there is any signal at all.

Runs on the unaugmented train split by default since that's where the null
results came from. --aug switches to the speed+VTLP cache.

    python src/iemocap/student/tune_kd_hparams.py
    python src/iemocap/student/tune_kd_hparams.py --aug --tag _aug
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    EPOCHS, load_inputs, load_teacher_signals, normalizer,
)
from iemocap.student.train_student import AUG_CACHE, METHODS, run  # noqa: E402

OUT_DIR = IEMOCAP_OUTPUTS / "student"
FEATURE_METHOD = "feature_kd_lasttoken"     # the target the probe preferred
FULL_METHOD = "full_kd_lasttoken"


def sweep_point(name, lam_logit, lam_feat, feat_key, t_kd, seed, data, epochs, student,
                lam_rkd=0.0, select="final"):
    """drop a temporary METHODS entry in and run it."""
    METHODS[name] = (lam_logit, lam_feat, feat_key)
    t0 = time.time()
    r = run(name, seed, data, epochs, t_kd=t_kd, student=student,
            lam_rkd=lam_rkd, select=select)
    r.update({"lam_logit": lam_logit, "lam_feature": lam_feat,
              "lam_rkd": lam_rkd,
              "feat_key": feat_key or "-", "seconds": round(time.time() - t0, 1)})
    return r


def main():
    ap = argparse.ArgumentParser(description="Greedy KD hyperparameter sweep (val-selected).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--student", default="small", choices=["small", "strong"])
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--lam-logits", type=float, nargs="+", default=[0.25, 0.5, 2.0, 4.0])
    ap.add_argument("--lam-feats", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--lam-rkds", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                    help="Relational-KD weights (Park et al. 2019: distance + angle)")
    ap.add_argument("--select", choices=["final", "val_max"], default="final",
                    help="final = fixed last epoch, no checkpoint selection")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    Xtr, ytr, ids_tr = load_inputs("train")
    Xva, yva, _ = load_inputs("val")
    Xte, yte, _ = load_inputs("test")
    mu, sd = normalizer(Xtr)
    teach = load_teacher_signals(ids_tr)
    if args.aug:
        aug = torch.load(AUG_CACHE, weights_only=False)
        teach = {k: v[aug["orig_idx"]] for k, v in teach.items()}
        Xtr, ytr = aug["X"], aug["labels"]
        mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, Xva, yva, Xte, yte, mu, sd, teach)
    print(f"student={args.student} n_train={len(Xtr)} aug={args.aug} seed={args.seed}\n")

    rows = []
    csv = OUT_DIR / f"sweep_kd_hparams{args.tag}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def record(stage, r):
        r["stage"] = stage
        r["n_train"] = len(Xtr)
        r["augmented"] = args.aug
        rows.append(r)
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"  [{stage}] T={r['kd_t']:<5g} lam_l={r['lam_logit']:<5g} "
              f"lam_f={r['lam_feature']:<5g}  ep={r['best_epoch']:3d}  "
              f"val UA={r['val_ua']:.4f}  test UA={r['test_ua']:.4f}  "
              f"({r['seconds']:.0f}s)", flush=True)

    # baseline for reference, no KD hyperparameter touches it
    record("baseline", sweep_point("ce_only", 0.0, 0.0, None, 1.0, args.seed,
                                   data, args.epochs, args.student, select=args.select))

    print("\n--- stage 1: temperature (logit-KD, lambda_logit=1) ---")
    for t in args.temps:
        record("1_temp", sweep_point("logit_kd", 1.0, 0.0, None, t, args.seed,
                                     data, args.epochs, args.student, select=args.select))
    s1 = pd.DataFrame([r for r in rows if r["stage"] == "1_temp"])
    best_t = float(s1.loc[s1.val_ua.idxmax(), "kd_t"])
    print(f"  -> best T by val UA: {best_t:g}")

    print(f"\n--- stage 2: lambda_logit (T={best_t:g}) ---")
    for lam in args.lam_logits:
        record("2_lam_logit", sweep_point("logit_kd", lam, 0.0, None, best_t, args.seed,
                                          data, args.epochs, args.student, select=args.select))
    s2 = pd.DataFrame([r for r in rows if r["stage"] in ("1_temp", "2_lam_logit")])
    s2 = s2[s2.kd_t == best_t]
    best_ll = float(s2.loc[s2.val_ua.idxmax(), "lam_logit"])
    print(f"  -> best lambda_logit by val UA: {best_ll:g}")

    print(f"\n--- stage 3: lambda_feature ({FEATURE_METHOD}) ---")
    feat_key = METHODS[FEATURE_METHOD][2]
    for lam in args.lam_feats:
        record("3_lam_feat", sweep_point(FEATURE_METHOD, 0.0, lam, feat_key, 1.0,
                                         args.seed, data, args.epochs, args.student, select=args.select))
    s3 = pd.DataFrame([r for r in rows if r["stage"] == "3_lam_feat"])
    best_lf = float(s3.loc[s3.val_ua.idxmax(), "lam_feature"])
    print(f"  -> best lambda_feature by val UA: {best_lf:g}")

    print(f"\n--- stage 4: combined (T={best_t:g}, lam_l={best_ll:g}, lam_f={best_lf:g}) ---")
    record("4_full", sweep_point(FULL_METHOD, best_ll, best_lf, feat_key, best_t,
                                 args.seed, data, args.epochs, args.student, select=args.select))

    # relational KD (Park et al. 2019): match the pairwise distance ratios and
    # the triplet angles of the teacher embedding instead of absolute positions.
    # worth a try for two reasons. both terms are invariant to shift and scale,
    # so the shared direction that eats ~42% of the cosine target drops out
    # instead of the student spending a batchnorm bias on it. and relational
    # objectives are supposed to hold up better when teacher and student differ
    # this much in capacity (4.7 B vs 96 K)
    print(f"\n--- stage 5: relational KD (target {feat_key}) ---")
    for lam in args.lam_rkds:
        record("5_rkd", sweep_point("ce_only", 0.0, 0.0, None, 1.0, args.seed,
                                    data, args.epochs, args.student,
                                    lam_rkd=lam, select=args.select))
    s5 = pd.DataFrame([r for r in rows if r["stage"] == "5_rkd"])
    best_rkd = float(s5.loc[s5.val_ua.idxmax(), "lam_rkd"])
    print(f"  -> best lambda_rkd by val UA: {best_rkd:g}")

    print(f"\n--- stage 6: logit + RKD (T={best_t:g}, lam_l={best_ll:g}, "
          f"lam_rkd={best_rkd:g}) ---")
    record("6_logit_rkd", sweep_point("logit_kd", best_ll, 0.0, None, best_t, args.seed,
                                      data, args.epochs, args.student,
                                      lam_rkd=best_rkd, select=args.select))

    df = pd.DataFrame(rows)
    base_val = float(df[df.stage == "baseline"].val_ua.iloc[0])
    base_test = float(df[df.stage == "baseline"].test_ua.iloc[0])
    df["val_ua_gain"] = (df.val_ua - base_val).round(4)
    df["test_ua_gain"] = (df.test_ua - base_test).round(4)
    df.to_csv(csv, index=False)

    print(f"\n=== CE baseline: val {base_val:.4f}  test {base_test:.4f} ===")
    for stage in ("1_temp", "2_lam_logit", "3_lam_feat", "4_full", "5_rkd", "6_logit_rkd"):
        s = df[df.stage == stage]
        if len(s):
            print(f"\n{stage}")
            print(s[["kd_t", "lam_logit", "lam_feature", "lam_rkd", "best_epoch",
                     "val_ua", "val_ua_gain", "test_ua", "test_ua_gain"]].to_string(index=False))
    print(f"\nCSV -> {csv}")
    print("Read the SHAPE, not the argmax: an ordered trend means a real effect "
          "off its peak; scatter means the sweep found nothing.")


if __name__ == "__main__":
    main()
