"""
Three summaries derived from run tables that already exist, so the numbers
quoted in FINDINGS.md come out of code rather than out of a notebook.

    loso_channel_decomposition.csv  what each channel is worth on its own.
        The two-stage arm uses a feature loss in stage 1 and a logit loss in the
        readout, so it never says which one pays. Reading the same encoders
        under a readout with no soft labels separates them.

    loso_screen_summary.csv         every fold-1 screen in one table, with the
        standard error of the five-seed mean, because most of the differences
        being compared are smaller than the spread across seeds.

    loso_target_vs_student.csv      the target's own k-NN against what the
        student trained on it scores. Within one family of targets the k-NN sets
        the ceiling and student-side fidelity adds nothing; a label-trained
        target then falls about two points below that ceiling, so the two are
        needed together.

    python src/iemocap/analysis/channel_decomposition.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_OUTPUTS  # noqa: E402

STUDENT = IEMOCAP_OUTPUTS / "student"
OUT = IEMOCAP_OUTPUTS / "analysis" / "teacher_compare"

# (label, method, readout). CE was not re-run for hubert, so both teachers are
# compared against the same baseline rows.
ARMS = [("CE encoder + CE head", "ce", "linear"),
        ("logit-KD encoder + CE head", "logit_kd", "linear"),
        ("logit-KD encoder + KD head", "logit_kd", "mlp_kd"),
        ("feature encoder + kNN", "feature_only_audio", "knn"),
        ("feature encoder + CE head", "feature_only_audio", "linear"),
        ("feature encoder + KD head", "feature_only_audio", "mlp_kd")]
SCREENS = [("target_variants.csv", "method"), ("lowdim_target.csv", "variant"),
           ("nuisance_target.csv", "variant"), ("reach_teacher_space.csv", "method")]


def per_fold(f, method, readout):
    p = STUDENT / f
    if not p.exists():
        return None
    s = pd.read_csv(p)
    s = s[s.protocol.str.startswith("loso") & (s.method == method) & (s.readout == readout)]
    return s.groupby("protocol").test_ua.mean() if len(s) else None


def channels():
    base = per_fold("stage2_readout.csv", "ce", "linear")
    rows = []
    for teacher, f in (("qwen", "stage2_readout.csv"), ("hubert", "stage2_readout_hubert.csv")):
        for label, m, r in ARMS:
            a = per_fold(f, m, r)
            if a is None:
                continue
            d = a.values - base.values
            rows.append({"teacher": teacher, "arm": label, "folds": len(a),
                         "test_ua": round(float(a.mean()), 4),
                         "delta_vs_ce_pp": round(float(d.mean() * 100), 2),
                         "folds_better": int((d > 0).sum()),
                         "p_paired": round(float(stats.ttest_rel(a.values, base.values).pvalue), 4)
                         if label != ARMS[0][0] else np.nan,
                         **{f"fold{p[4:]}": round(v, 4) for p, v in a.items()}})
    return pd.DataFrame(rows)


def screens():
    rows = []
    for f, key in SCREENS:
        p = STUDENT / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        # reach_teacher_space scores two heads, so its column is named for one
        if "test_ua" not in d and "mlp_test_ua" in d:
            d = d.rename(columns={"mlp_test_ua": "test_ua"})
        d = d[d.protocol == "loso1"] if "protocol" in d else d
        grp = [key] + [c for c in ("epochs", "readout") if c in d]
        for name, g in d.groupby(grp, dropna=False):
            rows.append({"screen": f.replace(".csv", ""),
                         "variant": name if isinstance(name, str) else "/".join(map(str, name)),
                         "n": len(g), "test_ua": round(float(g.test_ua.mean()), 4),
                         "sd": round(float(g.test_ua.std(ddof=1)), 4),
                         "se": round(float(g.test_ua.std(ddof=1) / np.sqrt(len(g))), 4),
                         "r2_train": round(float(g.r2_train.mean()), 4) if "r2_train" in g else np.nan,
                         "r2_test": round(float(g.r2_test.mean()), 4) if "r2_test" in g else np.nan,
                         "target_knn": round(float(g.knn_ua.mean()), 4) if "knn_ua" in g else np.nan})
    return pd.DataFrame(rows).sort_values("test_ua", ascending=False)


def target_vs_student(screen_df):
    """only the arms that kept SpecAugment and the 70-epoch schedule, so the
    target is the one thing that varies."""
    d = screen_df.dropna(subset=["target_knn"])
    d = d[~d.variant.str.contains("noaug") & ~d.variant.str.endswith("/210")]
    r_knn = stats.pearsonr(d.target_knn, d.test_ua)
    r_fid = stats.pearsonr(d.r2_train, d.test_ua)
    slope = np.polyfit(d.target_knn, d.test_ua, 1)[0]
    # does fidelity add anything once target quality is held fixed
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
    r_par = stats.pearsonr(res(d.r2_train.values, d.target_knn.values),
                           res(d.test_ua.values, d.target_knn.values))
    return d, pd.DataFrame([
        {"relation": "target k-NN vs student test UA", "n": len(d),
         "pearson_r": round(r_knn.statistic, 3), "p": round(r_knn.pvalue, 4),
         "slope": round(float(slope), 3)},
        {"relation": "student R2 train vs student test UA", "n": len(d),
         "pearson_r": round(r_fid.statistic, 3), "p": round(r_fid.pvalue, 4), "slope": np.nan},
        {"relation": "R2 train vs test UA, target k-NN partialled out", "n": len(d),
         "pearson_r": round(r_par.statistic, 3), "p": round(r_par.pvalue, 4), "slope": np.nan}])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ch = channels()
    sc = screens()
    d, rel = target_vs_student(sc)
    d.to_csv(OUT / "loso_target_vs_student.csv", index=False)
    for df, name in ((ch, "loso_channel_decomposition.csv"),
                     (sc, "loso_screen_summary.csv"),
                     (rel, "loso_target_vs_student_fit.csv")):
        df.to_csv(OUT / name, index=False)
        print(f"-> {OUT / name}")
    print("\n=== what each channel is worth (5 folds, paired, vs CE + CE head) ===")
    print(ch[["teacher", "arm", "test_ua", "delta_vs_ce_pp", "folds_better", "p_paired"]]
          .to_string(index=False))
    print("\n=== fold 1 screens, five seeds each ===")
    print(sc.head(20).to_string(index=False))
    print("\n=== what predicts the student ===")
    print(rel.to_string(index=False))


if __name__ == "__main__":
    main()
