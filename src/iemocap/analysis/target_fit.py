"""
Does the student fail to learn the teacher mapping, or fail to generalise it?

Feature-KD asks the student to reproduce, from log-mels, the 64-d vector the
teacher made for the same utterance. If it did that exactly its test accuracy
would be the probe's, 0.786 UA, not the 0.58 it actually gets. So something is
lost, and there are only two places to lose it:

    never fit the target at all             -> capacity
    fit train but the mapping didn't carry  -> generalisation

Those look the same in the accuracy column and completely different here,
because the teacher 64-d vectors exist for test too, they're just never used as
a training signal. Measuring cos(student z, teacher z) separately on train and
test tells the two apart in one number.

Three objectives, same architecture and schedule:

    ce          no teacher signal
    ce+cos      CE + cosine feature loss, what every feature-KD run used
    cos_only    CE weight zero, pure feature imitation

cos_only needs care: with no CE the classifier head gets no gradient and stays
at init, so its own logits mean nothing. So every method also gets scored with
a logistic regression head fitted on the train embeddings, which is the only
comparison that's fair to all three.

    python src/iemocap/analysis/target_fit.py
    python src/iemocap/analysis/target_fit.py --seeds 42 43 44 --epochs 70
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.repr_analysis import cosine_separability  # noqa: E402
from iemocap.analysis.cluster_features import teacher_reps  # noqa: E402
from iemocap.paths import IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LR, N_CLASSES, PROJ_DIM,
    SMALL_KW, WEIGHT_DECAY, DSResNetSE, kd_feature_loss, load_inputs,
    load_teacher_signals, normalizer,
)

OUT = IEMOCAP_OUTPUTS / "analysis"
SPLITS = ("train", "val", "test")
# (name, lam_ce, lam_feature)
METHODS = (("ce", 1.0, 0.0), ("ce+cos", 1.0, 1.0), ("cos_only", 0.0, 1.0))
FEAT_KEY = "audio_mean_l27"


def ua(y, p):
    return round(float(recall_score(y, p, average="macro",
                                    labels=list(range(N_CLASSES)), zero_division=0)), 4)


def train(lam_ce, lam_feat, data, epochs, seed):
    Xtr, ytr, mu, sd, t_z, evalsets = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=PROJ_DIM,
                       dropout=DROPOUT, **SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    for _ in range(epochs):
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


def main():
    ap = argparse.ArgumentParser(description="Where Feature-KD loses the teacher's mapping.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # teacher bottlenecks for every split. the test ones are for measurement
    # only, no student is trained against them
    _, teacher_z = teacher_reps(FEAT_KEY)

    # this is what makes target_cos readable. a student that gives up and emits
    # one constant vector for everything still scores well on a cosine loss,
    # because 19% of the target energy is a shared direction. the best such
    # constant is the mean of the normalised targets, so that is the floor
    collapse = {}
    for s in ("train", "test"):
        t = teacher_z[s][0]
        tn = t / np.linalg.norm(t, axis=1, keepdims=True)
        c = tn.mean(0)
        collapse[s] = round(float((tn @ (c / np.linalg.norm(c))).mean()), 4)
    print("collapse baseline (one constant vector for everything): "
          "train %.4f  test %.4f" % (collapse["train"], collapse["test"]), flush=True)

    Xtr, ytr, ids_tr = load_inputs("train")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    t_z = load_teacher_signals(ids_tr)["z_audio"]
    data = (Xtr, ytr, mu, sd, t_z, evalsets)

    rows = []
    for name, lam_ce, lam_feat in METHODS:
        for seed in args.seeds:
            print(f"{name} seed {seed} ...", flush=True)
            out = train(lam_ce, lam_feat, data, args.epochs, seed)

            ztr, ytr_np, _ = out["train"]
            clf = LogisticRegression(max_iter=2000)  # multinomial by default
            clf.fit(ztr, ytr_np)

            r = {"method": name, "seed": seed, "lam_ce": lam_ce, "lam_feature": lam_feat}
            for s in ("train", "test"):
                z, y, pred = out[s]
                tz = teacher_z[s][0]
                # how closely the student reproduces the teacher's actual vector
                r[f"target_cos_{s}"] = round(float(F.cosine_similarity(
                    torch.from_numpy(z), torch.from_numpy(tz), dim=1).mean()), 4)
                r[f"gap_{s}"] = round(cosine_separability(z, y)["gap"], 4)
                r[f"ua_head_{s}"] = ua(y, pred)
                r[f"ua_linprobe_{s}"] = ua(y, clf.predict(z))
            r["collapse_cos_train"] = collapse["train"]
            r["collapse_cos_test"] = collapse["test"]
            rows.append(r)
            print("   target cos train %.4f  test %.4f | test UA head %.4f  linprobe %.4f"
                  % (r["target_cos_train"], r["target_cos_test"],
                     r["ua_head_test"], r["ua_linprobe_test"]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "target_fit.csv", index=False)

    agg = df.groupby("method").agg(["mean", "std"]).round(4)
    cols = ["target_cos_train", "target_cos_test", "gap_train", "gap_test",
            "ua_head_test", "ua_linprobe_test"]
    print("\n=== mean over %d seeds ===" % len(args.seeds))
    print(agg[[(c, "mean") for c in cols]].to_string())
    print("\nsd:")
    print(agg[[(c, "std") for c in cols]].to_string())
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
