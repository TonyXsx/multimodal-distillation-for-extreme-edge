"""
What is actually inside the two teachers' soft labels, and what raising the
temperature does to it.

The student is trained against train-split logits, and the qwen head fits that
split at 96%, so at T=2 only a fifth of its probability mass sits off the target
class against a half for hubert. The obvious remedy is a larger T. But
temperature cannot create information: it rescales an ordering that is already
fixed, so the question is whether the ordering is worth amplifying. That is what
the structure block measures.

Written per fold, appended, so run it once per protocol.

    for f in 1 2 3 4 5; do IEMOCAP_PROTOCOL=loso$f python \
        src/iemocap/analysis/logit_channel_diagnostics.py; done

Two csvs:
    loso_logit_temperature.csv  the sweep, plus the T at which qwen's non-target
                                mass matches hubert's at T=2
    loso_logit_structure.csv    whether the non-target ranking carries anything
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_OUTPUTS, PROTOCOL  # noqa: E402

OUT = IEMOCAP_OUTPUTS / "analysis" / "teacher_compare"
TEMPS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
TEACHERS = ("qwen", "hubert")
REF_T = 2.0          # what every arm in the fixed protocol actually used
N_CLASSES = 4


def bank(teacher, split):
    """logits and labels from a teacher's LoRA feature cache."""
    tsuf = "" if teacher == "qwen" else f"_{teacher}"
    root = IEMOCAP_DATA / f"teacher_features{tsuf}_{PROTOCOL}"
    d = sorted(p for p in root.glob("*LORA*") if p.is_dir())[-1]
    b = torch.load(d / f"{split}_features.pt", weights_only=False, map_location="cpu")
    return b["features"]["logits"].float().numpy(), b["labels"].numpy()


def soft(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nontarget(p, y):
    """mass off the true class, and the renormalised distribution over the
    three wrong ones. The wrong classes are reindexed relative to the true
    class so rows of different classes are comparable."""
    n = len(y)
    m = np.ones_like(p, dtype=bool)
    m[np.arange(n), y] = False
    q = p[m].reshape(n, N_CLASSES - 1)
    return 1 - p[np.arange(n), y], q / np.maximum(q.sum(1, keepdims=True), 1e-12)


def match_temperature(logits, y, target_mass, lo=0.5, hi=64.0):
    """the T at which this teacher's mean non-target mass equals target_mass."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if nontarget(soft(logits, mid), y)[0].mean() < target_mass:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def eta2(z, g):
    """share of each column's variance explained by group membership."""
    mu, out = z.mean(0), np.zeros(z.shape[1])
    for u in np.unique(g):
        m = g == u
        out += m.sum() * (z[m].mean(0) - mu) ** 2
    return out / len(z) / np.maximum(z.var(0), 1e-12)


def confusion_alignment(q, y, te_logits, te_y):
    """does the dark knowledge on train predict the head's real confusions on
    unseen speakers? Correlates the mean non-target distribution per true class
    against the row-normalised off-diagonal of the test confusion matrix."""
    cm = confusion_matrix(te_y, te_logits.argmax(1), labels=range(N_CLASSES)).astype(float)
    a, b = [], []
    for c in range(N_CLASSES):
        off = np.delete(cm[c], c)
        if off.sum() == 0:
            continue
        a.append(off / off.sum())
        b.append(q[y == c].mean(0))
    a, b = np.concatenate(a), np.concatenate(b)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    if not PROTOCOL.startswith("loso"):
        raise SystemExit(f"loso folds only, got {PROTOCOL}")
    OUT.mkdir(parents=True, exist_ok=True)
    fold = int(PROTOCOL[4:])

    tr = {t: bank(t, "train") for t in TEACHERS}
    te = {t: bank(t, "test") for t in TEACHERS}
    # everything is compared against what hubert hands the student at T=2
    ref_mass = float(nontarget(soft(tr["hubert"][0], REF_T), tr["hubert"][1])[0].mean())

    temp_rows, struct_rows = [], []
    matched = {}
    for t in TEACHERS:
        lg, y = tr[t]
        matched[t] = match_temperature(lg, y, ref_mass)
        for T in TEMPS + (round(matched[t], 3),):
            p = soft(lg, T)
            mass, _ = nontarget(p, y)
            ent = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
            temp_rows.append({
                "fold": fold, "teacher": t, "T": T,
                "is_matched_T": T == round(matched[t], 3),
                "pmax": round(float(p.max(1).mean()), 4),
                "nontarget_mass": round(float(mass.mean()), 4),
                "entropy_nats": round(float(ent.mean()), 4),
                "entropy_norm": round(float(ent.mean() / np.log(N_CLASSES)), 4),
                "head_train_acc": round(float((lg.argmax(1) == y).mean()), 4),
                "head_test_acc": round(float((te[t][0].argmax(1) == te[t][1]).mean()), 4)})

        # structure of the ordering, at the shared T and at the matched one
        for tag, T in (("T2", REF_T), ("T_matched", matched[t])):
            q = nontarget(soft(lg, T), y)[1]
            other = TEACHERS[1 - TEACHERS.index(t)]
            qo = nontarget(soft(tr[other][0], REF_T), tr[other][1])[1]
            top, topo = q.argmax(1), qo.argmax(1)
            rho = [stats.spearmanr(a, b).statistic for a, b in zip(q[:2000], qo[:2000])]
            struct_rows.append({
                "fold": fold, "teacher": t, "at": tag, "T": round(T, 3),
                "nontarget_mass": round(float(nontarget(soft(lg, T), y)[0].mean()), 4),
                # how much of "which wrong class" the true class explains
                "nt_eta2": round(float(eta2(q, y).mean()), 4),
                # does it predict the head's real confusions on unseen speakers
                "nt_conf_r": round(confusion_alignment(q, y, *te[t]), 4),
                # do the two teachers nominate the same runner-up
                "nt_top_agree": round(float((top == topo).mean()), 4),
                "nt_top_kappa": round(float(cohen_kappa_score(top, topo)), 4),
                "nt_rank_rho": round(float(np.nanmean(rho)), 4)})

    for rows, name in ((temp_rows, "loso_logit_temperature.csv"),
                       (struct_rows, "loso_logit_structure.csv")):
        df = pd.DataFrame(rows)
        f = OUT / name
        if f.exists():
            old = pd.read_csv(f)
            df = pd.concat([old[old.fold != fold], df], ignore_index=True)
        df.sort_values(["fold", "teacher"]).to_csv(f, index=False)
        print(f"-> {f}")

    print(f"\nfold {fold}: hubert non-target mass at T=2 is {ref_mass:.4f}; "
          f"qwen needs T={matched['qwen']:.2f} to match it")
    print(pd.DataFrame(struct_rows).to_string(index=False))


if __name__ == "__main__":
    main()
