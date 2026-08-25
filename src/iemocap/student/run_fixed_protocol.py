"""
The fixed IEMOCAP measurement protocol, settled on 2026-08-22.

Everything before this used val-selected checkpoints, one to three seeds, and a
config that drifted between batches. The old numbers are archived under
outputs/iemocap/student/exploratory/ rather than deleted, because they are the
evidence that the measurement was the problem, not the method.

What is fixed here, and why each choice was made before seeing its result:

    SD split          the only protocol whose val is a decent proxy for test
                      (val - test = -0.5pp, against +2.9 to +5.7pp on SI).
                      Speaker leakage doesn't inflate the student - CE gets
                      0.5511 on SD and 0.5517 on SI - it only inflates the
                      teacher. So the measurement gets trustworthy without the
                      student's score moving.
    fixed 70 epochs   no checkpoint selection at all. Selecting on val pushes
                      the criterion-flip noise to 6.8pp and, over four batches,
                      hands CE 0.9-1.4pp more than it hands KD, so dropping it
                      is both stricter and fairer.
    5 seeds, paired   every method runs seeds 42-46, paired against the CE rows
                      with the same seed and the same config fingerprint. That
                      is what makes a ~1pp effect readable against a 6.2pp
                      between-config spread.
    val and test      both always recorded. val is for hyperparameter choice,
                      test is the headline, neither ever picks an epoch.

Results append to one master csv, so CE never needs re-running - later methods
pair against the stored CE rows as long as config_hash matches. The hash covers
the settings every method shares, so changing one of those shows up as a new
group instead of quietly corrupting the comparison. The lambdas and the feature
target are recorded as plain columns and left out of the hash.

Student embeddings are cached to data/iemocap/student{_sd}/z_cache/, so any
later question about the readout can be answered without retraining.

Every method gets scored twice, they aren't comparable otherwise:

    head       the model's own classifier. meaningless for feature_only, whose
               classifier gets no gradient and stays at init.
    linprobe   logistic regression on the train embeddings. the only readout
               that treats all three objectives equally.

    IEMOCAP_PROTOCOL=sd python src/iemocap/student/run_fixed_protocol.py --methods ce
    IEMOCAP_PROTOCOL=sd python src/iemocap/student/run_fixed_protocol.py \
        --methods feature_kd feature_only
    IEMOCAP_PROTOCOL=sd python src/iemocap/student/run_fixed_protocol.py --summary-only
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.models.audio_student import DSResNetSE  # noqa: E402
from common.losses import kd_feature_loss, kd_logit_loss  # noqa: E402
from iemocap.paths import PROTOCOL, IEMOCAP_OUTPUTS, IEMOCAP_STUDENT  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, CLASSES, DEVICE, DROPOUT, EPOCHS, FEATURE_TARGETS, LABEL_SMOOTH,
    LR, N_CLASSES, PROJ_DIM, SMALL_KW, WEIGHT_DECAY,
    available_splits, load_inputs, load_teacher_signals, normalizer,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "fixed_protocol_runs.csv"
ZCACHE = IEMOCAP_STUDENT / "z_cache"
SUMMARY_CSV = OUT / "fixed_protocol_summary.csv"
# resolved at run time. the LOSO folds have no val set, so the val_* columns
# just stay empty on those rows
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]

# the target goes in the name from here on. the first three keep their old names
# because rows for them already exist in fixed_protocol_runs.csv, and the
# feature_target column tells them apart.
#
# T = 2 is inherited from the archived SD batch on this same protocol, not
# re-selected. that gives logit-KD the benefit of the earlier tuning and the
# two-stage methods none - with lam_ce = 0 the cosine term is the only loss, and
# under AdamW rescaling the only gradient cancels out of m / sqrt(v), so
# lam_feature isn't a tunable knob at all
METHODS = {
    "ce":                 dict(ce=1.0, logit=0.0, feat=0.0, target=None,         T=None),
    "logit_kd":           dict(ce=1.0, logit=1.0, feat=0.0, target=None,         T=2.0),
    "feature_kd":         dict(ce=1.0, logit=0.0, feat=1.0, target="lasttoken",  T=None),
    "feature_kd_audio":   dict(ce=1.0, logit=0.0, feat=1.0, target="audio",      T=None),
    "full_kd_lasttoken":  dict(ce=1.0, logit=1.0, feat=1.0, target="lasttoken",  T=2.0),
    "full_kd_audio":      dict(ce=1.0, logit=1.0, feat=1.0, target="audio",      T=2.0),
    "feature_only":       dict(ce=0.0, logit=0.0, feat=1.0, target="lasttoken",  T=None),
    "feature_only_audio": dict(ce=0.0, logit=0.0, feat=1.0, target="audio",      T=None),
}


def metrics(y, p, prefix):
    labels = list(range(N_CLASSES))
    return {
        f"{prefix}_wa": round(float(accuracy_score(y, p)), 4),
        f"{prefix}_ua": round(float(recall_score(y, p, average="macro",
                                                 labels=labels, zero_division=0)), 4),
        f"{prefix}_macro_f1": round(float(f1_score(y, p, average="macro",
                                                   labels=labels, zero_division=0)), 4),
        f"{prefix}_weighted_f1": round(float(f1_score(y, p, average="weighted",
                                                      labels=labels, zero_division=0)), 4),
    }


def config_fingerprint():
    """the settings every method shares - protocol, architecture, optimiser,
    augmentation, reporting rule. rows only pair if the fingerprint matches, so
    a setting changed by accident can't look like a method effect.

    The lambdas, feature target and temperature are recorded as columns but not
    hashed. They are what makes a method a method, so hashing them would put
    each one in its own group with nothing left to pair against.
    """
    cfg = {"protocol": PROTOCOL, "student": "small", "epochs": EPOCHS, "lr": LR,
           "weight_decay": WEIGHT_DECAY, "batch": BATCH_SIZE, "dropout": DROPOUT,
           "label_smooth": LABEL_SMOOTH, "proj_dim": PROJ_DIM,
           "channels": list(SMALL_KW["channels"]), "proj_hidden": SMALL_KW["proj_hidden"],
           "spec_augment": True, "wave_augment": False, "select": "fixed_final_epoch",
           "readout": "standardised_logreg"}
    blob = json.dumps(cfg, sort_keys=True)
    return cfg, hashlib.sha1(blob.encode()).hexdigest()[:10]


def run_one(spec, t_z, t_logits, data, seed):
    lam_ce, lam_logit, lam_feat, T = spec["ce"], spec["logit"], spec["feat"], spec["T"]
    Xtr, ytr, mu, sd, evalsets = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=PROJ_DIM,
                       dropout=DROPOUT, **SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = spec_augment(((Xtr[idx].float() - mu) / sd).unsqueeze(1).to(DEVICE))
            z, logits = model(xb)
            loss = xb.new_zeros(())
            if lam_ce > 0:
                loss = loss + lam_ce * ce(logits, ytr[idx].to(DEVICE))
            if lam_logit > 0:
                loss = loss + lam_logit * kd_logit_loss(logits, t_logits[idx].to(DEVICE), T)
            if lam_feat > 0:
                loss = loss + lam_feat * kd_feature_loss(z, t_z[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for s, (Xs, ys) in evalsets.items():
            zs, ps = [], []
            for i in range(0, len(Xs), 256):
                xb = ((Xs[i:i + 256].float() - mu) / sd).unsqueeze(1).to(DEVICE)
                z, lg = model(xb)
                zs.append(z.cpu())
                ps.append(lg.argmax(1).cpu())
            out[s] = (torch.cat(zs).numpy(), ys.numpy(), torch.cat(ps).numpy())
    return out


def summarise():
    if not RUNS_CSV.exists():
        print("no runs yet")
        return
    from scipy import stats
    df = pd.read_csv(RUNS_CSV)
    # LOSO has no val, so those columns are missing on those rows
    keys = [k for k in ("test_ua", "test_wa", "test_macro_f1", "val_ua",
                        "linprobe_test_ua", "linprobe_val_ua") if k in df.columns]
    rows = []
    for h, grp in df.groupby("config_hash"):
        base = grp[grp.method == "ce"]
        for m, g in grp.groupby("method"):
            r = {"config_hash": h, "protocol": g.protocol.iloc[0],
                 "feature_target": g.feature_target.iloc[0],
                 "method": m, "n_seeds": len(g)}
            for k in keys:
                r[f"{k}_mean"] = round(float(g[k].mean()), 4)
                r[f"{k}_sd"] = round(float(g[k].std(ddof=1)) if len(g) > 1 else 0.0, 4)
            if m != "ce" and len(base):
                # pair on seed, only seeds present on both sides count
                j = g.merge(base, on="seed", suffixes=("", "_ce"))
                for k in ("test_ua", "linprobe_test_ua"):
                    d = j[k] - j[f"{k}_ce"]
                    r[f"{k}_delta"] = round(float(d.mean()), 4)
                    if len(d) > 1:
                        t, p = stats.ttest_rel(j[k], j[f"{k}_ce"])
                        r[f"{k}_t"] = round(float(t), 2)
                        r[f"{k}_p"] = round(float(p), 4)
                    r[f"{k}_n_paired"] = len(d)
            rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(SUMMARY_CSV, index=False)
    show = [c for c in ("method", "n_seeds", "val_ua_mean", "val_ua_sd",
                        "test_ua_mean", "test_ua_sd",
                        "linprobe_test_ua_mean", "linprobe_test_ua_sd") if c in s]
    print("\n=== fixed protocol (%s), own classifier head + linear probe ===" % PROTOCOL.upper())
    print(s[show].to_string(index=False))
    pcols = [c for c in ("test_ua_delta", "test_ua_p", "linprobe_test_ua_delta",
                         "linprobe_test_ua_p", "test_ua_n_paired") if c in s]
    if pcols:
        print("\n=== paired against the stored CE baseline (same seeds) ===")
        print(s[s.method != "ce"][["method"] + pcols].to_string(index=False))
    print("\n-> %s" % SUMMARY_CSV)


def main():
    ap = argparse.ArgumentParser(description="IEMOCAP student under the fixed protocol.")
    ap.add_argument("--methods", nargs="+", default=["ce"], choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarise()
        return

    print(f"protocol={PROTOCOL.upper()}  epochs={EPOCHS} (fixed)  seeds={args.seeds}")
    Xtr, ytr, ids_tr = load_inputs("train")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)
    teach = load_teacher_signals(ids_tr)
    print("  ".join(f"{k} {tuple(v[0].shape)}" for k, v in evalsets.items()))

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for name in args.methods:
        spec = METHODS[name]
        tgt = spec["target"]
        feat_key = FEATURE_TARGETS[tgt] if tgt else "none"
        cfg, h = config_fingerprint()
        t_z = teach[f"z_{tgt}"] if tgt else None
        t_logits = teach["logits"] if spec["logit"] > 0 else None

        for seed in args.seeds:
            if len(existing) and ((existing.method == name) & (existing.seed == seed)
                                  & (existing.config_hash == h)).any():
                print(f"  {name} seed {seed}: already in {RUNS_CSV.name}, skipping")
                continue
            t0 = time.time()
            out = run_one(spec, t_z, t_logits, data, seed)

            ztr, ytr_np, _ = out["train"]
            # standardise, otherwise lbfgs hits its iteration cap on the
            # feature_only embeddings, which is the row that depends on it
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000)).fit(ztr, ytr_np)

            # lam_* and kd_t are recorded but kept out of config_hash on
            # purpose, see config_fingerprint
            r = {"method": name, "seed": seed, "lam_ce": spec["ce"],
                 "lam_logit": spec["logit"], "lam_feature": spec["feat"],
                 "kd_t": spec["T"], "feature_target": feat_key, "config_hash": h, **cfg,
                 "seconds": round(time.time() - t0, 1)}
            for s in SPLITS:
                z, y, pred = out[s]
                r.update(metrics(y, pred, s))
                r.update(metrics(y, clf.predict(z), f"linprobe_{s}"))
            new.append(r)
            v = f"val UA {r['val_ua']:.4f}  " if "val_ua" in r else ""
            print(f"  {name} seed {seed}: {v}test UA {r['test_ua']:.4f}"
                  f"  | linprobe test {r['linprobe_test_ua']:.4f}"
                  f"  ({r['seconds']:.0f}s)", flush=True)

            # keep the embeddings, redoing a readout shouldn't cost a retrain
            ZCACHE.mkdir(parents=True, exist_ok=True)
            torch.save({s: {"z": out[s][0], "y": out[s][1], "pred": out[s][2]}
                        for s in SPLITS},
                       ZCACHE / f"{name}_seed{seed}_{h}.pt")
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(
                RUNS_CSV, index=False)

    summarise()


if __name__ == "__main__":
    main()
