"""
Rolls the five LOSO folds up into the headline table.

The unit here is the fold, not the seed. Seeds get averaged within a fold
first, then the paired t-test runs across the five folds against that fold's
own CE. That's the right pairing because what needs controlling for is how hard
the held-out session is, and the folds differ a lot on that (CE from 0.5013 to
0.5506). Pairing on seeds would treat fold difficulty as noise and throw away
most of the power.

It's also why LOSO settles what the single splits couldn't. On one SI split the
same method came out at +2.24pp, p = 0.085, and the variance there was from the
split rather than the init, so no number of seeds would have fixed it. Five
folds leave the effect size about the same and take p to 0.0002.

Absolute numbers don't compare to the single-split SI/SD studies. LOSO trains
on four sessions (~4,400) against SI's three (3,205) and averages over folds
that include the hardest session. Only the per-fold delta against CE carries
between studies.

    python src/iemocap/analysis/loso_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ROOT = _SRC.parent
OUT = ROOT / "outputs" / "iemocap" / "student"
FOLDS = [f"loso{k}" for k in range(1, 6)]

# label -> (encoder, stage-2 readout, or None for the jointly trained head)
METHODS = [
    ("2-stage, audio + mlp_kd",      "feature_only_audio", "mlp_kd"),
    ("2-stage, last_token + mlp_kd", "feature_only",       "mlp_kd"),
    ("full-KD, audio (T=2)",         "full_kd_audio",      None),
    ("full-KD, last_token (T=2)",    "full_kd_lasttoken",  None),
    ("logit-KD (T=2)",               "logit_kd",           None),
    ("feature-KD, audio",            "feature_kd_audio",   None),
    ("feature-KD, last_token",       "feature_kd",         None),
    ("CE (end-to-end)",              "ce",                 None),
]


def main():
    runs = pd.read_csv(OUT / "fixed_protocol_runs.csv")
    st2 = pd.read_csv(OUT / "stage2_readout.csv")

    def per_fold(enc, readout):
        """seed-averaged test UA per fold, in FOLDS order."""
        out = []
        for f in FOLDS:
            g = (runs[(runs.protocol == f) & (runs.method == enc)] if readout is None
                 else st2[(st2.protocol == f) & (st2.method == enc)
                          & (st2.readout == readout)])
            if g.empty:
                raise RuntimeError(f"no rows for {enc}/{readout} in {f}")
            out.append(g.test_ua.mean())
        return np.array(out)

    ce = per_fold("ce", None)
    rows, long = [], []
    for label, enc, ro in METHODS:
        v = per_fold(enc, ro)
        d = v - ce
        r = {"method": label, "readout": "own head" if ro is None else ro,
             "mean_test_ua": round(v.mean(), 4), "fold_sd": round(v.std(ddof=1), 4),
             "n_folds": len(v), "seeds_per_fold": 5}
        if enc != "ce":
            t, p = stats.ttest_rel(v, ce)
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            r.update({"delta_pp": round(d.mean() * 100, 2),
                      "ci95_lo_pp": round((d.mean() - half) * 100, 2),
                      "ci95_hi_pp": round((d.mean() + half) * 100, 2),
                      "paired_t": round(float(t), 2), "p": round(float(p), 4),
                      "folds_positive": int((d > 0).sum())})
        rows.append(r)
        for f, ua, dd in zip(FOLDS, v, d):
            long.append({"method": label, "fold": f, "test_ua": round(ua, 4),
                         "delta_pp": round(dd * 100, 2)})

    tab = pd.DataFrame(rows).sort_values("mean_test_ua", ascending=False)
    tab.to_csv(OUT / "loso_main_table.csv", index=False)
    pd.DataFrame(long).to_csv(OUT / "loso_per_fold.csv", index=False)

    print("=== 5-fold LOSO, 5 seeds per fold, paired across folds (n=5) ===")
    print(tab.to_string(index=False))
    print("\n=== per-fold test UA ===")
    print(pd.DataFrame(long).pivot(index="method", columns="fold",
                                   values="test_ua").reindex(
                                       [m[0] for m in METHODS]).to_string())
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
