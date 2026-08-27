"""
Can the student land in the teacher's space, rather than near its direction.

The two-stage encoder is trained with a cosine loss, which fixes direction and
leaves scale and offset free, and its output comes from a bare linear
projection while the teacher target comes out of LayerNorm followed by GELU. So
the student is never asked for, and cannot naturally produce, the vector the
teacher actually emits. That shows up twice: the fidelity anomaly in
teacher_compare/FINDINGS.md §11, where the student output has a mean norm of 1.3
on train and 25.8 on test, and the fact that nothing so far tests whether the
teacher's own probe head would work on a student embedding.

Two factors, four cells, one fold, five seeds:

    loss    cosine, as now, or MSE onto the raw post-GELU target
    head    the bare projection, or the same LayerNorm + GELU the probe uses

`cos x bare` is the current method and reproduces it. Everything else is the
fixed protocol unchanged: lambda_ce = 0 in stage 1, 70 epochs, no checkpoint
selection, seeds 42-46.

Each encoder is then scored two ways:

    mlp_kd     the trained stage-2 readout, comparable with every earlier table
    probehead  the teacher's own frozen 64->4 probe head, applied directly.
               This one only works if the student is in the teacher's space and
               its units, so it is the actual test of the question.

Fidelity is reported as a vector R-squared, 1 - SSE over the variance around the
train mean, which unlike the cosine fit does not move when the student rescales
or shifts its output.

    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/reach_teacher_space.py
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.losses import kd_feature_loss, kd_logit_loss  # noqa: E402
from common.models.audio_student import DSResNetSE  # noqa: E402
from common.probe import Probe  # noqa: E402
from iemocap.paths import PROTOCOL, IEMOCAP_DATA, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LR, N_CLASSES, PROJ_DIM,
    SMALL_KW, WEIGHT_DECAY, available_splits, load_inputs, normalizer,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "reach_teacher_space.csv"
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]
FEAT_KEY = "audio_mean_l27"
HEAD_EPOCHS, HEAD_LR, HEAD_T = 3000, 1e-2, 2.0

# (loss, match the probe block on the student side)
METHODS = {
    "cos_bare": ("cos", False),      # the current method
    "cos_lngelu": ("cos", True),
    "mse_bare": ("mse", False),
    "mse_lngelu": ("mse", True),
}


class MatchHead(nn.Module):
    """LayerNorm + GELU, the same block the probe bottleneck comes out of, so
    the student can produce vectors that live where the teacher's do."""

    def __init__(self, match):
        super().__init__()
        self.ln = nn.LayerNorm(PROJ_DIM) if match else None

    def forward(self, z):
        return F.gelu(self.ln(z)) if self.ln is not None else z


def metrics(y, p, prefix):
    labels = list(range(N_CLASSES))
    return {f"{prefix}_wa": round(float(accuracy_score(y, p)), 4),
            f"{prefix}_ua": round(float(recall_score(y, p, average="macro",
                                                     labels=labels, zero_division=0)), 4),
            f"{prefix}_macro_f1": round(float(f1_score(y, p, average="macro",
                                                       labels=labels, zero_division=0)), 4)}


def teacher_side():
    """the 64-d post-GELU target for both splits, the head that reads it, and
    the logits the stage-2 readout uses."""
    feat = glob.glob(str(IEMOCAP_DATA / f"teacher_features_{PROTOCOL}" / "*LORA*"))[0]
    pdir = IEMOCAP_DATA / f"teacher_probe_{PROTOCOL}" / "bottleneck" / "adapted" / FEAT_KEY
    ck = torch.load(pdir / "checkpoint.pt", weights_only=False, map_location="cpu")
    m = Probe(ck["in_dim"], [PROJ_DIM], N_CLASSES, dropout=0.1).eval()
    m.load_state_dict(ck["state_dict"])
    out = {}
    for s in SPLITS:
        d = torch.load(f"{feat}/{s}_features.pt", weights_only=False, map_location="cpu")
        X = (d["features"][FEAT_KEY].float() - ck["mu"]) / ck["sd"]
        with torch.no_grad():
            _, z = m(X, return_bottleneck=True)
        out[s] = {"z": z, "y": d["labels"].numpy(),
                  "ids": d["sample_ids"], "logits": d["features"]["logits"].float()}
    return out, m.head.to(DEVICE).eval()


def run_one(loss_kind, match, target, data, seed):
    Xtr, ytr, mu, sd, evalsets = data
    t_z = target["train"]["z"].to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=PROJ_DIM,
                       dropout=DROPOUT, **SMALL_KW).to(DEVICE)
    head = MatchHead(match).to(DEVICE)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = spec_augment(((Xtr[idx].float() - mu) / sd).unsqueeze(1).to(DEVICE))
            z = head(model(xb)[0])
            zt = t_z[idx]
            loss = F.mse_loss(z, zt) if loss_kind == "mse" else kd_feature_loss(z, zt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    head.eval()
    out = {}
    with torch.no_grad():
        for s, (Xs, ys) in evalsets.items():
            zs = []
            for i in range(0, len(Xs), 256):
                xb = ((Xs[i:i + 256].float() - mu) / sd).unsqueeze(1).to(DEVICE)
                zs.append(head(model(xb)[0]).cpu())
            out[s] = (torch.cat(zs), ys.numpy())
    return out


def stage2(ztr, ytr, t_logits, seed):
    """the mlp_kd readout from stage2_readout.py, unchanged."""
    torch.manual_seed(seed)
    mu, sd = ztr.mean(0, keepdims=True), ztr.std(0, keepdims=True) + 1e-6
    X = torch.from_numpy((ztr - mu) / sd).float().to(DEVICE)
    y = torch.from_numpy(ytr).long().to(DEVICE)
    net = nn.Sequential(nn.Linear(X.shape[1], PROJ_DIM), nn.ReLU(),
                        nn.Linear(PROJ_DIM, N_CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    tl = t_logits.to(DEVICE)
    for _ in range(HEAD_EPOCHS):
        net.train()
        opt.zero_grad()
        o = net(X)
        (lossf(o, y) + kd_logit_loss(o, tl, HEAD_T)).backward()
        opt.step()
        sched.step()
    net.eval()

    def predict(z):
        with torch.no_grad():
            return net(torch.from_numpy((z - mu) / sd).float().to(DEVICE)).argmax(1).cpu().numpy()
    return predict


def fidelity(zs, zt, mu_tr):
    """vector R-squared against the train mean, plus the cosine numbers for
    continuity with the earlier tables."""
    sse = float(((zs - zt) ** 2).sum())
    sst = float(((zt - mu_tr) ** 2).sum())
    u = zs / zs.norm(dim=1, keepdim=True)
    v = zt / zt.norm(dim=1, keepdim=True)
    c = v.mean(0)
    base = float((v @ (c / c.norm())).mean())
    cos = float((u * v).sum(1).mean())
    return {"r2": round(1 - sse / sst, 3), "cos": round(cos, 4),
            "fit_pct": round((cos - base) / (1 - base) * 100, 1),
            "norm_ratio": round(float(zs.norm(dim=1).mean() / zt.norm(dim=1).mean()), 3)}


def main():
    ap = argparse.ArgumentParser(description="Does the student reach the teacher's space.")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    target, probe_head = teacher_side()
    Xtr, ytr, ids = load_inputs("train")
    if list(target["train"]["ids"]) != list(ids):
        raise RuntimeError("teacher feature ids do not match the log-mel cache ids")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)
    mu_tr = target["train"]["z"].mean(0, keepdim=True)
    print(f"protocol={PROTOCOL}  seeds={args.seeds}  target {tuple(mu_tr.shape)}", flush=True)

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for name in args.methods:
        loss_kind, match = METHODS[name]
        for seed in args.seeds:
            t0 = time.time()
            out = run_one(loss_kind, match, target, data, seed)
            ztr = out["train"][0].numpy()
            predict = stage2(ztr, out["train"][1], target["train"]["logits"], seed)
            r = {"protocol": PROTOCOL, "method": name, "loss": loss_kind,
                 "head": "ln_gelu" if match else "bare", "seed": seed,
                 "seconds": round(time.time() - t0, 1)}
            for s in SPLITS:
                z, y = out[s]
                r.update(metrics(y, predict(z.numpy()), f"mlp_{s}"))
                with torch.no_grad():
                    p = probe_head(z.to(DEVICE)).argmax(1).cpu().numpy()
                r.update(metrics(y, p, f"probehead_{s}"))
                r.update({f"{k}_{s}": v for k, v in
                          fidelity(z, target[s]["z"], mu_tr).items()})
            new.append(r)
            print(f"  {name} seed {seed}: mlp {r['mlp_test_ua']:.4f}  "
                  f"probehead {r['probehead_test_ua']:.4f}  "
                  f"R2 train {r['r2_train']} test {r['r2_test']}  "
                  f"|z|/|zt| {r['norm_ratio_test']}  ({r['seconds']:.0f}s)", flush=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(
                RUNS_CSV, index=False)

    df = pd.read_csv(RUNS_CSV)
    print("\n=== mean over seeds ===")
    print(df.groupby(["loss", "head"]).agg(
        n=("seed", "size"), mlp_ua=("mlp_test_ua", "mean"), mlp_sd=("mlp_test_ua", "std"),
        probehead_ua=("probehead_test_ua", "mean"),
        r2_train=("r2_train", "mean"), r2_test=("r2_test", "mean"),
        fit_test=("fit_pct_test", "mean"), norm_ratio=("norm_ratio_test", "mean")
    ).round(4).to_string())
    print(f"\n-> {RUNS_CSV}")


if __name__ == "__main__":
    main()
