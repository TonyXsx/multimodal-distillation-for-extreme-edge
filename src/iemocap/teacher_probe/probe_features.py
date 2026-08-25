"""
Probes the extracted IEMOCAP teacher features, frozen arm against adapted.

Doing two things at once.

First the control: train the same probe on every 2048-d feature from both arms
and compare. That is what decides whether the LoRA step was worth it and where
it was worth it. On MIntRec the answer was lopsided - the frozen audio feature
probed at 0.5443 and adaptation only moved it to 0.5533, while the
transcript-conditioned readout got 0.6130. So nearly all the benefit landed in
a representation an audio-only student cannot reach. IEMOCAP might come out
differently, since the audio tower was adapted properly here (47.2 M LoRA
params at rank 64 vs MIntRec 7.9 M on q/k/v only) and emotion does live in
prosody.

Second the feature-KD targets. The probe is 2048 -> 64 -> 4, so the 64-d
bottleneck lines up with the student projection directly, no extra projector
needed. Those activations get saved for train and val.

Test bottlenecks are not saved, on purpose. The student never gets a teacher
signal on test, same rule as FSC and MIntRec. Test features are read here only
to report how each representation generalises, which is a statement about the
teacher rather than something handed to the student.

Protocol is fixed across every probe so the comparison stays clean, same as the
FSC B2 probe: 50 epochs, AdamW(1e-3, wd 1e-4), batch 256, dropout 0.1, CE,
standardised on train stats, no early stopping and no selection on val.

logits is skipped as a probe input since it is the 4-d prediction, not a
representation. Its argmax accuracy is reported separately as the teacher
readout, which doubles as a check that the extraction agrees with what
eval_teacher.py measured on the model itself.

    python src/iemocap/teacher_probe/probe_features.py
    python src/iemocap/teacher_probe/probe_features.py --features last_token audio_mean_l27
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.probe import Probe  # noqa: E402
from common.training import DEVICE  # noqa: E402
from iemocap.paths import IEMOCAP_OUTPUTS, IEMOCAP_PROBE, find_adapted_features  # noqa: E402
from iemocap.teacher.data import CLASSES  # noqa: E402

ARMS = {"adapted": None}   # resolved at run time. frozen is optional, see --arms
SPLITS = ("train", "val", "test")

# same as the FSC B2 probe
EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, DROPOUT, BOTTLENECK, SEED = 50, 1e-3, 1e-4, 256, 0.1, 64, 42

OUT_ROOT = IEMOCAP_PROBE / "bottleneck"
OUT_CSV = IEMOCAP_OUTPUTS / "teacher_probe"


def load_arm(arm):
    d = find_adapted_features()
    out = {}
    for s in SPLITS:
        f = d / f"{s}_features.pt"
        if not f.exists():                # LOSO has no val
            continue
        r = torch.load(f, weights_only=False)
        out[s] = {"features": r["features"], "labels": r["labels"], "ids": r["sample_ids"]}
    return out


def metrics(y, p):
    labels = list(range(len(CLASSES)))
    return {
        "wa": round(float(accuracy_score(y, p)), 4),
        "ua": round(float(recall_score(y, p, average="macro", labels=labels, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=labels, zero_division=0)), 4),
    }


def train_probe(Xtr, ytr, in_dim, n_classes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = Probe(in_dim, [BOTTLENECK], n_classes, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def infer(model, X, batch=512):
    model.eval()
    preds, zs = [], []
    for i in range(0, X.shape[0], batch):
        logits, z = model(X[i:i + batch].to(DEVICE), return_bottleneck=True)
        preds.append(logits.argmax(1).cpu())
        zs.append(z.cpu())
    return torch.cat(preds), torch.cat(zs)


def main():
    ap = argparse.ArgumentParser(description="Probe IEMOCAP teacher features (frozen vs adapted).")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--features", nargs="+", default=None, help="default: every 2048-d key")
    ap.add_argument("--no-save", action="store_true", help="skip writing bottleneck targets")
    args = ap.parse_args()

    rows = []
    for arm in args.arms:
        data = load_arm(arm)
        keys = args.features or sorted(k for k, v in data["train"]["features"].items()
                                       if v.shape[1] > len(CLASSES))
        ytr = data["train"]["labels"]

        # the teacher readout, for reference and to cross-check the extraction
        if "logits" in data["train"]["features"]:
            for s in (s for s in ("val", "test") if s in data):
                lg = data[s]["features"]["logits"].float()
                rows.append({"arm": arm, "feature": "logits(argmax)", "split": s,
                             "n": len(lg), **metrics(data[s]["labels"].numpy(),
                                                     lg.argmax(1).numpy())})

        for key in keys:
            Xtr = data["train"]["features"][key].float()
            mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
            model = train_probe((Xtr - mu) / sd, ytr, Xtr.shape[1], len(CLASSES))

            saved = {}
            present = [s for s in SPLITS if s in data]   # LOSO has no val
            for s in present:
                Xs = (data[s]["features"][key].float() - mu) / sd
                pred, z = infer(model, Xs)
                saved[s] = z
                if s != "train":
                    rows.append({"arm": arm, "feature": key, "split": s,
                                 "n": len(pred), **metrics(data[s]["labels"].numpy(), pred.numpy())})
                else:
                    rows.append({"arm": arm, "feature": key, "split": "train",
                                 "n": len(pred), **metrics(ytr.numpy(), pred.numpy())})
            print(f"  {arm:8s} {key:26s} "
                  + "  ".join(f"{r['split']}:UA={r['ua']:.4f}"
                              for r in rows[-len(present):]), flush=True)

            if not args.no_save:
                d = OUT_ROOT / arm / key
                d.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                            "in_dim": Xtr.shape[1], "bottleneck": BOTTLENECK,
                            "classes": CLASSES}, d / "checkpoint.pt")
                # never test, the student gets no teacher signal on held-out data
                torch.save({s: {"z": saved[s], "labels": data[s]["labels"],
                                "ids": data[s]["ids"]}
                            for s in present if s != "test"},
                           d / "bottleneck_reps.pt")

    df = pd.DataFrame(rows)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV / "probe_results.csv", index=False)

    if (df.split == "val").any():          # LOSO has no val
        print("\n=== val UA by arm x feature ===")
        piv = df[df.split == "val"].pivot(index="feature", columns="arm", values="ua")
        if {"adapted", "frozen"} <= set(piv.columns):
            piv["delta"] = (piv["adapted"] - piv["frozen"]).round(4)
        print(piv.to_string())
    print("\n=== test UA by arm x feature ===")
    print(df[df.split == "test"].pivot(index="feature", columns="arm", values="ua").to_string())
    print(f"\nCSV -> {OUT_CSV / 'probe_results.csv'}")
    if not args.no_save:
        print(f"Feature-KD targets -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
