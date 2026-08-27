"""
Stage 2. How much does the readout on top of a frozen student embedding matter?

feature_only trains its 96k encoder with no labels at all, only the teacher 64-d
vector, so its own classifier head never gets a gradient and means nothing. It
gets scored with a head fitted on the train embeddings and applied unchanged to
val and test. Nothing is ever fitted on test.

That two-stage recipe is not the same as feature-KD. The difference is where CE
is allowed to act:

    feature_kd    CE and cosine optimised together, so CE gradients go through
                  the whole encoder and shape z
    feature_only  the encoder is shaped by cosine alone, CE only ever touches
                  the 260-parameter readout

target_fit.csv showed that isn't cosmetic - taking CE out of the encoder cost
6pp of fit on train but tripled how much of the teacher mapping survived to test
(7.4% -> 25.4%).

So the remaining question is whether the reported number depends on which
readout is used. This refits several heads on the cached embeddings. Nothing is
retrained, so all five seeds are free:

    logreg      standardised logistic regression, what fixed_protocol_runs used
    linear      nn.Linear(64, 4), the head the network itself carries, trained
                with the same CE and label smoothing
    mlp         64 -> 64 -> 4, does a non-linear readout find more
    knn         k=10 cosine kNN, parameter free, a sanity floor
    ffn         64 -> 256 -> 64 -> 4, the expand-then-contract shape of a
                transformer FFN block, 33,348 params
    linear_kd   linear plus the teacher logit-KD term at T=2
    mlp_kd      mlp plus the same
    ffn_kd      ffn plus the same

Each _kd head is paired with its plain twin deliberately. mlp_kd beat every
linear readout, but it changed two things at once, width and loss, and the 2x2
showed neither does anything alone. Any new head shape has to be reported the
same way or the interaction gets credited to whichever factor gets mentioned
first.

If the first four agree then the readout isn't doing the work and the
feature_only advantage is a property of the representation.

The _kd pair exists because of something the fixed protocol turned up: refitting
a plain head on frozen features helps a CE-trained encoder (+0.89pp, p = 0.002)
but hurts a logit-KD one (-1.89pp, p = 0.032). So part of what logit-KD buys is
a better head, not a better representation, and a two-stage recipe that throws
that away in stage 2 is leaving it on the table. linear_kd puts it back - the
encoder still sees no labels, and stage 2 fits the head against both the labels
and the teacher soft targets.

    python src/iemocap/student/stage2_readout.py
    python src/iemocap/student/stage2_readout.py --readouts logreg linear
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import PROTOCOL, TSUF, IEMOCAP_OUTPUTS, IEMOCAP_STUDENT  # noqa: E402
from common.losses import kd_logit_loss  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    DEVICE, LABEL_SMOOTH, N_CLASSES, WEIGHT_DECAY, load_inputs, load_teacher_signals,
)

OUT = IEMOCAP_OUTPUTS / "student"
ZCACHE = IEMOCAP_STUDENT / f"z_cache{TSUF}"
# resolved from the cached embeddings, the LOSO folds have no val set
SPLITS = ("train", "val", "test")
# 300 full-batch steps at lr 1e-3 left the linear head badly underfitted on the
# feature_only embeddings (0.45 vs 0.57 for a converged lbfgs). that looks like a
# property of the representation when it is really just an optimisation artefact
HEAD_EPOCHS, HEAD_LR = 3000, 1e-2


def metrics(y, p, prefix):
    labels = list(range(N_CLASSES))
    return {
        f"{prefix}_wa": round(float(accuracy_score(y, p)), 4),
        f"{prefix}_ua": round(float(recall_score(y, p, average="macro",
                                                 labels=labels, zero_division=0)), 4),
        f"{prefix}_macro_f1": round(float(f1_score(y, p, average="macro",
                                                   labels=labels, zero_division=0)), 4),
    }


def torch_head(ztr, ytr, hidden=None, seed=0, t_logits=None, T=2.0, lam_logit=1.0):
    """head trained the way the network's own head would have been. CE with the
    same label smoothing, AdamW at the same lr/wd, full batch, cosine anneal.

    Pass t_logits to add the teacher logit-KD term on top, at the same T=2
    everything else here uses."""
    torch.manual_seed(seed)
    mu, sd = ztr.mean(0, keepdims=True), ztr.std(0, keepdims=True) + 1e-6
    X = torch.from_numpy((ztr - mu) / sd).float().to(DEVICE)
    y = torch.from_numpy(ytr).long().to(DEVICE)
    # hidden is None, one width, or a tuple. (256, 64) gives the
    # expand-then-contract shape of a transformer FFN block
    dims = [] if hidden is None else ([hidden] if isinstance(hidden, int) else list(hidden))
    layers, d = [], X.shape[1]
    for h in dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, N_CLASSES)]
    head = nn.Sequential(*layers).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    tl = torch.from_numpy(t_logits).float().to(DEVICE) if t_logits is not None else None
    for _ in range(HEAD_EPOCHS):
        head.train()
        opt.zero_grad()
        out = head(X)
        loss = lossf(out, y)
        if tl is not None:
            loss = loss + lam_logit * kd_logit_loss(out, tl, T)
        loss.backward()
        opt.step()
        sched.step()
    head.eval()

    def predict(z):
        with torch.no_grad():
            xb = torch.from_numpy((z - mu) / sd).float().to(DEVICE)
            return head(xb).argmax(1).cpu().numpy()
    return predict


def build(readout, ztr, ytr, seed, t_logits=None):
    if readout == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(ztr, ytr)
        return clf.predict
    if readout == "knn":
        clf = make_pipeline(StandardScaler(),
                            KNeighborsClassifier(n_neighbors=10, metric="cosine")).fit(ztr, ytr)
        return clf.predict
    if readout == "linear":
        return torch_head(ztr, ytr, hidden=None, seed=seed)
    if readout == "mlp":
        return torch_head(ztr, ytr, hidden=64, seed=seed)
    if readout == "linear_kd":
        return torch_head(ztr, ytr, hidden=None, seed=seed, t_logits=t_logits)
    if readout == "mlp_kd":
        return torch_head(ztr, ytr, hidden=64, seed=seed, t_logits=t_logits)
    if readout == "ffn":
        return torch_head(ztr, ytr, hidden=(256, 64), seed=seed)
    if readout == "ffn_kd":
        return torch_head(ztr, ytr, hidden=(256, 64), seed=seed, t_logits=t_logits)
    raise ValueError(readout)


def main():
    ap = argparse.ArgumentParser(description="Refit readouts on cached student embeddings.")
    ap.add_argument("--readouts", nargs="+",
                    default=["knn", "logreg", "linear", "mlp", "ffn",
                             "linear_kd", "mlp_kd", "ffn_kd"])
    ap.add_argument("--methods", nargs="+", default=None, help="default: every cached method")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(ZCACHE.glob("*.pt"))
    if args.methods:
        files = [f for f in files if f.stem.split("_seed")[0] in args.methods]
    if not files:
        raise FileNotFoundError(f"no cached embeddings in {ZCACHE} -- run run_fixed_protocol.py")
    print(f"{len(files)} cached runs in {ZCACHE}")

    # teacher logits for the _kd readouts. the cache stores splits in the order
    # load_inputs returns them, which is what load_teacher_signals asserts
    # against, so the rows line up. checked below anyway
    t_logits = None
    if any(r.endswith("_kd") for r in args.readouts):
        _, ytr_ref, ids_tr = load_inputs("train")
        t_logits = load_teacher_signals(ids_tr)["logits"].numpy()
        print(f"teacher logits {t_logits.shape} loaded for the _kd readouts")

    rows = []
    for f in files:
        method, seedtag, _ = f.stem.split("_seed")[0], f.stem.split("_seed")[1], None
        seed = int(seedtag.split("_")[0])
        d = torch.load(f, weights_only=False)
        ztr, ytr = d["train"]["z"], d["train"]["y"]
        if t_logits is not None and not np.array_equal(ytr, ytr_ref.numpy()):
            raise RuntimeError(f"{f.name}: cached train labels do not match the manifest order")
        for ro in args.readouts:
            predict = build(ro, ztr, ytr, seed, t_logits)
            r = {"protocol": PROTOCOL, "method": method, "seed": seed, "readout": ro}
            for s in SPLITS:
                if s not in d:            # LOSO has no val
                    continue
                r.update(metrics(d[s]["y"], predict(d[s]["z"]), s))
            # the network own head, for reference. meaningless for feature_only
            r["ownhead_test_ua"] = round(float(np.mean([
                (d["test"]["pred"][d["test"]["y"] == c] == c).mean()
                for c in range(N_CLASSES)])), 4)
            rows.append(r)
        print(f"  {f.stem}: " + "  ".join(
            f"{r['readout']}={r['test_ua']:.4f}" for r in rows[-len(args.readouts):]), flush=True)

    df = pd.DataFrame(rows)
    # one file holds every protocol and the path has no protocol suffix, so a
    # plain overwrite would wipe the other protocol's whole grid. merge instead
    # and only replace the cells this run recomputed
    csv = OUT / f"stage2_readout{TSUF}.csv"
    if csv.exists():
        prev = pd.read_csv(csv)
        if "protocol" in prev.columns:
            key = ["protocol", "method", "readout", "seed"]
            idx = pd.MultiIndex.from_frame(df[key])
            keep = prev[~pd.MultiIndex.from_frame(prev[key]).isin(idx)]
            df = pd.concat([keep, df], ignore_index=True)
            print(f"merged: kept {len(keep)} existing rows, wrote {len(rows)} new")
    df.to_csv(csv, index=False)

    print("\n=== test UA by readout (mean over seeds) ===")
    print(df.pivot_table(index="method", columns="readout", values="test_ua",
                         aggfunc="mean").round(4).to_string())
    print("\nsd:")
    print(df.pivot_table(index="method", columns="readout", values="test_ua",
                         aggfunc=lambda x: x.std(ddof=1)).round(4).to_string())
    if "val_ua" in df.columns and df.val_ua.notna().any():   # LOSO has no val
        print("\n=== val UA by readout (mean over seeds) ===")
        print(df.pivot_table(index="method", columns="readout", values="val_ua",
                             aggfunc="mean").round(4).to_string())
    print("\n-> %s" % csv)


if __name__ == "__main__":
    main()
