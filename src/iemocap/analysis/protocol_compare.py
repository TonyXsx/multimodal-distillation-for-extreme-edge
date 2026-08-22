"""
Speaker-independent vs speaker-dependent: is the teacher's feature space
actually better under SD, or only easier to score?

The two protocols share everything -- same 5,531 utterances, same LoRA teacher,
same extraction code -- and differ only in how the utterances are split:

    SI   sessions 2,3,4 train | session 5 val | session 1 test  (no shared speakers)
    SD   stratified on speaker x emotion, 60/20/20              (all 10 speakers everywhere)

So any difference in the numbers below is caused by the split and nothing else.
Two separate things are worth telling apart, and the probe UA alone conflates
them:

    QUALITY      how separable the classes are inside each split
    CONSISTENCY  whether val and test agree with train, and with each other

A representation can score well on the first and badly on the second -- that is
exactly what SI does, and it is why val was a poor proxy for test there.

The 64-d row is the Feature-KD target. Its probe is retrained here per protocol
under the identical recipe used by teacher_probe/probe_features.py (50 epochs,
AdamW 1e-3/1e-4, batch 256, dropout 0.1, train-only standardisation), in memory
and without touching the saved SI checkpoints.

Outputs (outputs/iemocap/analysis/):
    protocol_compare.csv        every metric, per protocol x representation x split
    fig_protocol_compare.png    class separability, SI vs SD, per split

Usage:
    python src/iemocap/analysis/protocol_compare.py
    python src/iemocap/analysis/protocol_compare.py --features last_token
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.probe import Probe  # noqa: E402
from common.repr_analysis import (  # noqa: E402
    cosine_separability, knn_transfer, plot_grouped_bars, silhouette,
)
from common.training import DEVICE  # noqa: E402
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_OUTPUTS  # noqa: E402

OUT = IEMOCAP_OUTPUTS / "analysis"
SPLITS = ("train", "val", "test")
FEATS = ("audio_mean_l27", "last_token")
TAG = "iemocap4__qwen2.5-omni-3b-bf16-LORA__adapter_ep3__audio-tr"
# the two feature roots differ only by the split the teacher was tuned/extracted on
PROTOCOLS = {"SI": IEMOCAP_DATA / "teacher_features" / TAG,
             "SD": IEMOCAP_DATA / "teacher_features_sd" / TAG}

# identical to teacher_probe/probe_features.py -- do not drift from it
EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, DROPOUT, BOTTLENECK, SEED = 50, 1e-3, 1e-4, 256, 0.1, 64, 42


def load(root, key):
    out = {}
    for s in SPLITS:
        r = torch.load(root / f"{s}_features.pt", weights_only=False)
        out[s] = (r["features"][key].float(), r["labels"].long())
    return out


def fit_probe(Xtr, ytr, n_classes):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = Probe(Xtr.shape[1], [BOTTLENECK], n_classes, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            lossf(m(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE)).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def bottleneck(m, X, batch=512):
    zs, ps = [], []
    for i in range(0, len(X), batch):
        logits, z = m(X[i:i + batch].to(DEVICE), return_bottleneck=True)
        zs.append(z.cpu())
        ps.append(logits.argmax(1).cpu())
    return torch.cat(zs).numpy(), torch.cat(ps).numpy()


def score(reps, name, protocol, k, probe_ua=None):
    rows = []
    Ztr, ytr = reps["train"]
    for s in SPLITS:
        Z, y = reps[s]
        r = {"protocol": protocol, "representation": name, "split": s, "n": len(y),
             "dim": Z.shape[1], **cosine_separability(Z, y), "silhouette": silhouette(Z, y)}
        if s != "train":
            r.update({f"knn_{a}": b for a, b in knn_transfer(Ztr, ytr, Z, y, k=k).items()})
        if probe_ua is not None:
            r["probe_ua"] = probe_ua[s]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description="SI vs SD: teacher feature quality and consistency.")
    ap.add_argument("--features", nargs="+", default=list(FEATS))
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for protocol, root in PROTOCOLS.items():
        for key in args.features:
            raw = load(root, key)
            Xtr, ytr = raw["train"]
            mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
            std = {s: (((X - mu) / sd), y) for s, (X, y) in raw.items()}

            print(f"{protocol} {key}: probe ...", flush=True)
            probe = fit_probe(std["train"][0], ytr, int(ytr.max()) + 1)

            lo, ua = {}, {}
            for s in SPLITS:
                X, y = std[s]
                z, pred = bottleneck(probe, X)
                lo[s] = (z, y.numpy())
                # UA = macro recall, the metric used everywhere else in this project
                ua[s] = round(float(np.mean([
                    (pred[y.numpy() == c] == c).mean() for c in range(int(ytr.max()) + 1)])), 4)

            hi = {s: (X.numpy(), y.numpy()) for s, (X, y) in std.items()}
            rows += score(hi, f"{key} [2048]", protocol, args.k)
            rows += score(lo, f"{key} [64]", protocol, args.k, probe_ua=ua)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "protocol_compare.csv", index=False)

    bars = df.copy()
    bars["series"] = bars.protocol + " " + bars.representation
    plot_grouped_bars(bars, "gap", "split", "series", OUT / "fig_protocol_compare.png",
                      title="Class separability of the teacher's features, SI vs SD",
                      subtitle="Cosine gap (between-class minus within-class similarity). "
                               "Same teacher, same utterances, only the split differs.",
                      ylabel="cosine separability gap")

    print("\n=== quality: separability and silhouette ===")
    print(df.pivot_table(index=["representation", "protocol"], columns="split",
                         values="gap").reindex(columns=list(SPLITS)).round(4).to_string())
    print("\n=== consistency: k-NN transfer from train, and probe UA (64-d only) ===")
    sub = df[df.split != "train"]
    print(sub.pivot_table(index=["representation", "protocol"], columns="split",
                          values="knn_ua").round(4).to_string())
    print()
    p = df[df.probe_ua.notna()] if "probe_ua" in df else df.iloc[:0]
    if len(p):
        print(p.pivot_table(index=["representation", "protocol"], columns="split",
                            values="probe_ua").reindex(columns=list(SPLITS)).round(4).to_string())
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
