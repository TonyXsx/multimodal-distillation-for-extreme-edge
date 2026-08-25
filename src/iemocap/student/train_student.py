"""
Trains the tiny audio-only IEMOCAP student, with and without distillation.

Six conditions, several seeds each:

    ce_only                 labels only, the baseline
    logit_kd                + KL to the teacher 4-way head
    feature_kd_audio        + cosine to bottleneck(audio_mean_l27), clean target
    feature_kd_lasttoken    + cosine to bottleneck(last_token), privileged
    full_kd_audio           logit + feature, clean
    full_kd_lasttoken       logit + feature, privileged

Both feature targets get run because the probe results leave it genuinely
uncertain here, and in the opposite direction from MIntRec - on test the clean
audio feature probes slightly above the privileged readout (UA 0.7861 vs
0.7806). If that carries into the student then aligning a text-free student to
a text-shaped representation isn't the liability we assumed, at least once the
teacher audio tower has been adapted too.

Selection is on val UA, matching the teacher stage, because the splits have
different class priors. Test gets evaluated once per run using the val-selected
checkpoint. No run picks its epoch by test.

Everything except the loss is fixed across conditions: the tuned FSC recipe,
reused as is (70 epochs, AdamW 1e-3/1e-4, batch 128, cosine, label smoothing
0.1, SpecAugment, T=8, lambda 1.0/1.0). No new hyperparameter search here.

    python src/iemocap/student/train_student.py                       # everything
    python src/iemocap/student/train_student.py --methods ce_only --seeds 42 --epochs 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from iemocap.paths import IEMOCAP_OUTPUTS, IEMOCAP_STUDENT  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, CLASSES, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LAM_FEATURE,
    LAM_LOGIT, LR, N_CLASSES, PROJ_DIM, SMALL_KW, T_KD, WEIGHT_DECAY,
    DSResNetSE, kd_feature_loss, kd_logit_loss, rkd_loss, load_inputs,
    load_teacher_signals, model_summary, normalizer,
)

# the two capacities from the FSC 2x2 study, unchanged. strong is DSResNetSE
# defaults, small is the reduced channel plan. for 4 classes that is
# small 96,236 params / 0.37 MiB fp32, strong 377,748 / 1.44 MiB
STUDENTS = {
    "small":  {"channels": (16, 32, 64, 96, 128), "proj_hidden": None},
    "strong": {},
}

METHODS = {
    "ce_only":              (0.0, 0.0, None),
    "logit_kd":             (LAM_LOGIT, 0.0, None),
    "feature_kd_audio":     (0.0, LAM_FEATURE, "z_audio"),
    "feature_kd_lasttoken": (0.0, LAM_FEATURE, "z_lasttoken"),
    "full_kd_audio":        (LAM_LOGIT, LAM_FEATURE, "z_audio"),
    "full_kd_lasttoken":    (LAM_LOGIT, LAM_FEATURE, "z_lasttoken"),
}
OUT_DIR = IEMOCAP_OUTPUTS / "student"
AUG_CACHE = IEMOCAP_STUDENT / "logmel" / "train_aug.pt"


def metrics(y, p):
    lab = list(range(N_CLASSES))
    return {
        "wa": round(float(accuracy_score(y, p)), 4),
        "ua": round(float(recall_score(y, p, average="macro", labels=lab, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=lab, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, p, average="weighted", labels=lab, zero_division=0)), 4),
    }


@torch.no_grad()
def evaluate(model, X, y, mu, sd, batch=256):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = ((X[i:i + batch].float() - mu) / sd).unsqueeze(1).to(DEVICE)
        preds.append(model(xb)[1].argmax(1).cpu())
    return metrics(y.numpy(), torch.cat(preds).numpy())


def run(method, seed, data, epochs, t_kd=T_KD, student="small",
        lam_rkd=0.0, rkd_key="z_lasttoken", select="val_max"):
    """select decides which checkpoint the test numbers come from.

    "val_max" is the usual thing, the epoch with the highest val UA. On this
    dataset that is nearly useless: over 15 hyperparameter points, val UA at
    its own argmax correlates with test UA at r = 0.15, and the test spread it
    induces is 6.0 points against 3.8 for the final epoch. Val is 1,241
    utterances from two speakers, so a max over 70 noisy epochs is mostly
    picking luck.

    "final" just reports the last epoch, which removes that variance.
    """
    lam_logit, lam_feat, feat_key = METHODS[method]
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, yva, Xte, yte, mu, sd, teach = data
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=PROJ_DIM,
                       dropout=DROPOUT, **STUDENTS[student]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    t_logits = teach["logits"] if lam_logit else None
    t_z = teach[feat_key] if feat_key else None
    t_rkd = teach[rkd_key] if lam_rkd else None

    best_ua, best_state, best_ep = -1.0, None, -1
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:            # batchnorm needs more than 1
                continue
            xb = ((Xtr[idx].float() - mu) / sd).unsqueeze(1).to(DEVICE)
            xb = spec_augment(xb)
            yb = ytr[idx].to(DEVICE)
            z, logits = model(xb)
            loss = ce(logits, yb)
            if lam_logit:
                loss = loss + lam_logit * kd_logit_loss(logits, t_logits[idx].to(DEVICE), t_kd)
            if lam_feat:
                loss = loss + lam_feat * kd_feature_loss(z, t_z[idx].to(DEVICE))
            if lam_rkd:
                loss = loss + lam_rkd * rkd_loss(z, t_rkd[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        m = evaluate(model, Xva, yva, mu, sd)
        if m["ua"] > best_ua:
            best_ua, best_ep = m["ua"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # score the final-epoch model as well as the val-selected one. val here has
    # almost no class structure (silhouette ~0 for both CE and KD students) so
    # selecting on val UA is noisy, and it keeps landing on epochs 12-37 of 70
    # while the lr is still high. recording both shows whether a result comes
    # from the training or from the selection
    fin_val = evaluate(model, Xva, yva, mu, sd)
    fin_test = evaluate(model, Xte, yte, mu, sd)

    if select == "final":
        val_m, test_m = fin_val, fin_test
        best_ep = epochs
    else:
        model.load_state_dict(best_state)
        val_m = evaluate(model, Xva, yva, mu, sd)
        test_m = evaluate(model, Xte, yte, mu, sd)
    return {"method": method, "seed": seed, "student": student, "best_epoch": best_ep,
            "kd_t": t_kd, "lam_rkd": lam_rkd, "select": select,
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
            **{f"final_val_{k}": v for k, v in fin_val.items()},
            **{f"final_test_{k}": v for k, v in fin_test.items()}}


def main():
    ap = argparse.ArgumentParser(description="Train the tiny IEMOCAP audio-only student.")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--kd-t", type=float, default=T_KD,
                    help="softmax temperature for Logit-KD. The default 8 was tuned on "
                         "FSC's 31 classes; with 4 classes it flattens the teacher to "
                         "near-uniform (correct-class mass 0.40 against a 0.25 floor).")
    ap.add_argument("--center-feature", action="store_true",
                    help="subtract the train-set per-dimension mean from the teacher's "
                         "feature target before the cosine loss. ~42%% of each target "
                         "vector's direction is a component shared by every sample, and "
                         "the student's BatchNorm bottleneck cannot reproduce it, so it "
                         "is an unreachable constant rather than usable signal.")
    ap.add_argument("--aug", action="store_true",
                    help="use the speed+VTLP augmented training cache (train_aug.pt); "
                         "val and test are never augmented")
    ap.add_argument("--student", choices=list(STUDENTS), default="small",
                    help="capacity: the FSC 2x2 small (96K) or strong (378K) DSResNet-SE")
    ap.add_argument("--swap-val-test", action="store_true",
                    help="use Session 1 as validation and Session 5 as test, the reverse of "
                         "the default. Reported alongside the default orientation it measures "
                         "how much a single IEMOCAP fold decides the answer; reported alone it "
                         "would be fold-shopping, since Session 1 is already known to be the "
                         "harder of the two.")
    ap.add_argument("--tag", default="", help="suffix for the output CSV filenames")
    args = ap.parse_args()

    Xtr, ytr, ids_tr = load_inputs("train")
    va_split, te_split = ("test", "val") if args.swap_val_test else ("val", "test")
    Xva, yva, _ = load_inputs(va_split)
    Xte, yte, _ = load_inputs(te_split)
    print(f"orientation: val={va_split} ({len(Xva)}) test={te_split} ({len(Xte)})")
    mu, sd = normalizer(Xtr)
    teach = load_teacher_signals(ids_tr)

    if args.aug:
        # speed + VTLP copies of train. each copy inherits its source
        # utterance's teacher signals through orig_idx - the teacher heard the
        # original and its judgement doesn't change when the copy is faster or
        # has shifted formants. so CE gets N times the (input, label) pairs
        # while KD gets N times the (input, teacher output) pairs
        aug = torch.load(AUG_CACHE, weights_only=False)
        idx = aug["orig_idx"]
        teach = {k: v[idx] for k, v in teach.items()}
        Xtr, ytr = aug["X"], aug["labels"]
        mu, sd = normalizer(Xtr)
        print(f"augmented train: {len(Xtr)} from {int(idx.max()) + 1} utterances "
              f"(speeds {sorted(set(aug['speed'].tolist()))}, "
              f"VTLP {aug['vtlp'].min():.2f}-{aug['vtlp'].max():.2f})")

    if args.center_feature:
        for k in ("z_audio", "z_lasttoken"):
            teach[k] = teach[k] - teach[k].mean(0, keepdim=True)
    data = (Xtr, ytr, Xva, yva, Xte, yte, mu, sd, teach)

    summ = model_summary(DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES,
                                    proj_dim=PROJ_DIM, dropout=DROPOUT, **STUDENTS[args.student]))
    print(f"student[{args.student}]: {summ[chr(39)+chr(39)] if False else summ['params']:,} params  {summ['fp32_mb']:.3f} MiB fp32  "
          f"({summ['int8_mb']:.3f} MiB int8)")
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} | input {tuple(Xtr.shape[1:])}")
    print(f"teacher signals: logits{tuple(teach['logits'].shape)} "
          f"z_audio{tuple(teach[chr(39)+chr(39)] if False else teach['z_audio'].shape)}")

    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for seed in args.seeds:
            t0 = time.time()
            r = run(method, seed, data, args.epochs, t_kd=args.kd_t, student=args.student)
            r["seconds"] = round(time.time() - t0, 1)
            r["center_feature"] = bool(args.center_feature)
            r["val_split"], r["test_split"] = va_split, te_split
            r["n_train"], r["augmented"] = len(Xtr), bool(args.aug)
            rows.append(r)
            print(f"  {method:22s} seed={seed}  ep={r['best_epoch']:3d}  "
                  f"val UA={r['val_ua']:.4f}  test UA={r['test_ua']:.4f}  "
                  f"test UA={r[chr(39)+chr(39)] if False else r['test_ua']:.4f} -> final {r['final_test_ua']:.4f}  ({r['seconds']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"results{args.tag}.csv", index=False)

    df = pd.DataFrame(rows)
    agg = (df.groupby("method")[["val_ua", "test_ua", "test_macro_f1", "final_test_ua", "final_test_macro_f1"]]
             .agg(["mean", "std"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex([m for m in METHODS if m in set(df.method)])
    agg.to_csv(OUT_DIR / f"summary{args.tag}.csv")

    base = agg.loc["ce_only", "test_ua_mean"] if "ce_only" in agg.index else None
    if base is not None:
        agg["test_ua_gain"] = (agg["test_ua_mean"] - base).round(4)
    print("\n=== summary (mean over seeds) ===")
    print(agg.to_string())
    print(f"\nCSVs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
