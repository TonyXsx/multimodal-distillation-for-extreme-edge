"""
Boundary extension for the KD hyperparameter sweep (tune_kd_hparams.py).

The first sweep's optima all sat at the upper edge of their grids, so we probe
one step further to see if performance is still climbing or has plateaued.
Only TWO new runs (reusing the existing run()); existing CSV rows are NOT
deleted — the new rows are appended and the plot is regenerated from the full set.

  1. T-sweep extension : T=16 at lam_logit=0.5   (vs existing T=2/4/8 @0.5)
  2. lam_logit ext     : T=8, lam_logit=3.0       (vs existing 0.1/0.5/1.0 @T8)

(lam_feature is NOT extended — it already plateaued: 1.0->3.0 was only +0.1pp.)
"""

import csv
import sys
from pathlib import Path

PROJECT = Path(r"D:\msc_AI\individual_project\multimodal-distillation-for-extreme-edge")
sys.path.insert(0, str(PROJECT / "src" / "student"))
from tune_kd_hparams import run, make_plot, OUT      # noqa: E402  (reuse, no main() on import)
from train_student import load_data                  # noqa: E402

CSV = OUT / "kd_hparam_sweep.csv"


def main():
    # Load existing rows (don't delete) and coerce numeric fields.
    rows = []
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("val_acc", "macro_f1", "weighted_f1"):
                r[k] = float(r[k])
            rows.append(r)
    ce_f1 = next(r["macro_f1"] for r in rows if r["stage"] == "ce_only")

    def lookup(name):
        return next((r["macro_f1"] for r in rows if r["name"] == name), None)
    prev_T8 = lookup("T8_ll0.5")
    prev_ll1 = lookup("T8_ll1")

    train_data, val_data = load_data()

    new = [
        ("logit_T_sweep",  "T16_ll0.5", 16.0, 0.5, 0.0),   # extend T one step
        ("logit_ll_sweep", "T8_ll3",     8.0, 3.0, 0.0),   # extend lam_logit one step at T=8
    ]
    for stage, name, t, ll, lf in new:
        m = run(train_data, val_data, t=t, lam_logit=ll, lam_feature=lf)
        rows.append({"stage": stage, "name": name, "T": t, "lam_logit": ll, "lam_feature": lf,
                     "val_acc": round(m["acc"], 4), "macro_f1": round(m["macro_f1"], 4),
                     "weighted_f1": round(m["weighted_f1"], 4)})
        print(f"  [{stage}] {name}: acc={m['acc']:.4f} macroF1={m['macro_f1']:.4f} wF1={m['weighted_f1']:.4f}")

    # Append-safe rewrite (all old rows preserved + 2 new rows).
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "name", "T", "lam_logit", "lam_feature",
                                          "val_acc", "macro_f1", "weighted_f1"])
        w.writeheader(); w.writerows(rows)
    print(f"\nUpdated CSV -> {CSV}")

    t16 = next(r["macro_f1"] for r in rows if r["name"] == "T16_ll0.5")
    ll3 = next(r["macro_f1"] for r in rows if r["name"] == "T8_ll3")
    print("\n=== boundary check (macro-F1) ===")
    print(f"  T sweep @ll0.5 :  T8={prev_T8:.4f}  ->  T16={t16:.4f}   ({'still up' if t16 > prev_T8 else 'plateaued/down'})")
    print(f"  lam_logit @T8  :  ll1={prev_ll1:.4f} -> ll3={ll3:.4f}   ({'still up' if ll3 > prev_ll1 else 'plateaued/down'})")

    make_plot(rows, ce_f1)
    print("Done.")


if __name__ == "__main__":
    main()
