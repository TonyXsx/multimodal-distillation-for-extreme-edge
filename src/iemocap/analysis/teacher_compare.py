"""
Qwen against HuBERT, same folds, same seeds, same student.

The FSC control (chapter 4) put a frozen HuBERT beside the frozen Qwen and the
gap came out at 0.3-0.4pp on single runs, too small to call. This is the same
question asked where the protocol is strongest: five LOSO folds, five paired
seeds, and a student config whose fingerprint is identical between the two
arms.

CE is not re-run. Nothing in the CE objective touches a teacher tensor, so the
stored rows are the baseline for both arms, which also means the two teachers
are measured against exactly the same numbers rather than against two draws of
the same thing.

Two comparisons come out of it. Each teacher against that shared CE, paired
over folds, which is the loso_summary table repeated per arm. And the contrast
between the arms, fold by fold, which is the only line that says anything about
the teacher rather than about distillation.

Only the audio-target methods appear. HuBERT has no transcript-conditioned
readout to build a last_token target from.

    python src/iemocap/analysis/teacher_compare.py
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
    ("2-stage, audio + mlp_kd", "feature_only_audio", "mlp_kd"),
    ("full-KD, audio (T=2)",    "full_kd_audio",      None),
    ("logit-KD (T=2)",          "logit_kd",           None),
    ("feature-KD, audio",       "feature_kd_audio",   None),
]


def load(teacher):
    suf = "" if teacher == "qwen" else f"_{teacher}"
    return (pd.read_csv(OUT / f"fixed_protocol_runs{suf}.csv"),
            pd.read_csv(OUT / f"stage2_readout{suf}.csv"))


def per_fold(runs, st2, enc, readout):
    """seed-averaged test UA per fold, in FOLDS order."""
    out = []
    for f in FOLDS:
        g = (runs[(runs.protocol == f) & (runs.method == enc)] if readout is None
             else st2[(st2.protocol == f) & (st2.method == enc) & (st2.readout == readout)])
        if g.empty:
            raise RuntimeError(f"no rows for {enc}/{readout} in {f}")
        out.append(g.test_ua.mean())
    return np.array(out)


def paired(v, base):
    d = v - base
    t, p = stats.ttest_rel(v, base)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"delta_pp": round(d.mean() * 100, 2),
            "ci95_lo_pp": round((d.mean() - half) * 100, 2),
            "ci95_hi_pp": round((d.mean() + half) * 100, 2),
            "paired_t": round(float(t), 2), "p": round(float(p), 4),
            "folds_positive": int((d > 0).sum())}


def main():
    arms = {t: load(t) for t in ("qwen", "hubert")}
    ce = per_fold(*arms["qwen"], "ce", None)      # teacher-agnostic, shared

    rows, long, gains = [], [], {}
    for teacher, (runs, st2) in arms.items():
        for label, enc, ro in METHODS:
            v = per_fold(runs, st2, enc, ro)
            gains[(teacher, label)] = v - ce
            rows.append({"teacher": teacher, "method": label,
                         "readout": "own head" if ro is None else ro,
                         "mean_test_ua": round(v.mean(), 4),
                         "fold_sd": round(v.std(ddof=1), 4), **paired(v, ce)})
            for f, ua in zip(FOLDS, v):
                long.append({"teacher": teacher, "method": label, "fold": f,
                             "test_ua": round(ua, 4), "delta_pp": round((ua - ce[FOLDS.index(f)]) * 100, 2)})
    rows.append({"teacher": "shared", "method": "CE (end-to-end)", "readout": "own head",
                 "mean_test_ua": round(ce.mean(), 4), "fold_sd": round(ce.std(ddof=1), 4)})

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "teacher_compare.csv", index=False)
    pd.DataFrame(long).to_csv(OUT / "teacher_compare_per_fold.csv", index=False)

    # the arms differ only in the teacher, so their gains pair fold by fold
    contrast = []
    for label, _, _ in METHODS:
        d = gains[("qwen", label)] - gains[("hubert", label)]
        t, p = stats.ttest_rel(gains[("qwen", label)], gains[("hubert", label)])
        contrast.append({"method": label, "qwen_minus_hubert_pp": round(d.mean() * 100, 2),
                         "paired_t": round(float(t), 2), "p": round(float(p), 4),
                         "folds_qwen_ahead": int((d > 0).sum())})
    con = pd.DataFrame(contrast)
    con.to_csv(OUT / "teacher_contrast.csv", index=False)

    print("=== 5-fold LOSO, 5 seeds per fold, against the shared CE baseline ===")
    print(tab.sort_values("mean_test_ua", ascending=False).to_string(index=False))
    print("\n=== qwen gain minus hubert gain, paired over folds (n=5) ===")
    print(con.to_string(index=False))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
