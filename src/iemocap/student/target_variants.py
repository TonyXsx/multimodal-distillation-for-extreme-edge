"""
Five ways to hand the same teacher to the same student, one fold, five seeds.

The diagnostics in outputs/iemocap/analysis/teacher_compare/ say the 64-d target
is the problem rather than the teacher: 86% of its train-side variance is the
class label, a fifth of its energy is a shared direction that a constant vector
gets for free, and 95% of its sample-to-sample structure is explained by which
pair of classes two utterances belong to. This runs the three fixes that follow
from that, against the target we have been using.

    base        the current target, post-GELU from the 50-epoch probe
    A_perdim    pre-GELU from a 5-epoch probe, both sides centred per dimension
    A_persample the same target, both sides centred per sample instead
    B_pca       an unsupervised PCA-64 of the same 2048-d feature, no labels
    C_rkd2048   relational KD straight onto the raw 2048-d feature

Everything else is the fixed protocol unchanged: DSResNet-SE small, 70 epochs,
no checkpoint selection, seeds 42-46, and the stage-2 mlp_kd head copied from
stage2_readout.py. Stage 1 always has lambda_ce = 0, so this is the two-stage
scheme with only the target swapped, and `base` reproduces the run already in
fixed_protocol_runs.csv as the control on the reimplementation.

One fold at a time, on purpose. This is a screen, not a result.

    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/target_variants.py
    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/target_variants.py \
        --methods B_pca --seeds 42
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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, recall_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.losses import kd_feature_loss, kd_logit_loss, rkd_loss  # noqa: E402
from common.models.audio_student import DSResNetSE  # noqa: E402
from common.probe import Probe  # noqa: E402
from iemocap.paths import PROTOCOL, IEMOCAP_DATA, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LR, N_CLASSES, PROJ_DIM,
    SMALL_KW, WEIGHT_DECAY, available_splits, load_inputs, normalizer,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "target_variants.csv"
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]
FEAT_KEY = "audio_mean_l27"
HEAD_EPOCHS, HEAD_LR, HEAD_T = 3000, 1e-2, 2.0
PROBE_SEED = 42

# (target builder, centring, loss)
METHODS = {
    "base":        ("probe50_postgelu", "none",       "cos"),
    "A_perdim":    ("probe5_pregelu",   "perdim",     "cos"),
    "A_persample": ("probe5_pregelu",   "persample",  "cos"),
    "B_pca":       ("pca64",            "perdim",     "cos"),
    "C_rkd2048":   ("raw2048",          "none",       "rkd"),
}


def metrics(y, p, prefix):
    labels = list(range(N_CLASSES))
    return {f"{prefix}_wa": round(float(accuracy_score(y, p)), 4),
            f"{prefix}_ua": round(float(recall_score(y, p, average="macro",
                                                     labels=labels, zero_division=0)), 4),
            f"{prefix}_macro_f1": round(float(f1_score(y, p, average="macro",
                                                       labels=labels, zero_division=0)), 4)}


def teacher_features():
    """the standardised 2048-d feature both splits, plus the head logits."""
    feat = glob.glob(str(IEMOCAP_DATA / f"teacher_features_{PROTOCOL}" / "*LORA*"))[0]
    pdir = IEMOCAP_DATA / f"teacher_probe_{PROTOCOL}" / "bottleneck" / "adapted" / FEAT_KEY
    ck = torch.load(pdir / "checkpoint.pt", weights_only=False, map_location="cpu")
    out = {}
    for s in SPLITS:
        d = torch.load(f"{feat}/{s}_features.pt", weights_only=False, map_location="cpu")
        h = d["features"][FEAT_KEY].float()
        out[s] = {"h": (h - ck["mu"]) / ck["sd"], "y": d["labels"].numpy(),
                  "ids": d["sample_ids"], "logits": d["features"]["logits"].float()}
    return out, ck


def train_probe(X, y, epochs, seed=PROBE_SEED):
    """probe_features.py's recipe, stopped wherever the caller wants."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Probe(X.shape[1], [PROJ_DIM], N_CLASSES, dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    X, y = X.to(DEVICE), torch.from_numpy(y).long().to(DEVICE)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(m(X[idx]), y[idx]).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def probe_activations(m, X, pre_gelu):
    """the bottleneck, either side of its GELU."""
    lin, ln = m.blocks[0][0], m.blocks[0][1]
    pre = ln(lin(X.to(DEVICE)))
    return (pre if pre_gelu else F.gelu(pre)).cpu()


def build_target(kind, tf):
    """returns {split: tensor}. every statistic is fitted on train only."""
    Xtr = tf["train"]["h"]
    if kind == "raw2048":
        return {s: tf[s]["h"] for s in SPLITS}
    if kind == "pca64":
        p = PCA(n_components=PROJ_DIM, random_state=0).fit(Xtr.numpy())
        return {s: torch.from_numpy(p.transform(tf[s]["h"].numpy())).float() for s in SPLITS}
    epochs, pre = (50, False) if kind == "probe50_postgelu" else (5, True)
    m = train_probe(Xtr, tf["train"]["y"], epochs)
    return {s: probe_activations(m, tf[s]["h"], pre) for s in SPLITS}


def centre(z, mode, mu=None):
    """the same operation is applied to the student and the teacher."""
    if mode == "perdim":
        return z - (z.mean(0, keepdim=True) if mu is None else mu)
    if mode == "persample":
        return z - z.mean(1, keepdim=True)
    return z


def run_one(spec, target, data, seed):
    _, mode, loss_kind = spec
    Xtr, ytr, mu, sd, evalsets = data
    t_z = target["train"]
    t_mu = t_z.mean(0, keepdim=True)
    t_z = centre(t_z, mode, t_mu).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=PROJ_DIM,
                       dropout=DROPOUT, **SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = spec_augment(((Xtr[idx].float() - mu) / sd).unsqueeze(1).to(DEVICE))
            z, _ = model(xb)
            zt = t_z[idx]
            if loss_kind == "rkd":
                loss = rkd_loss(z, zt)
            else:
                loss = kd_feature_loss(centre(z, mode), zt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for s, (Xs, ys) in evalsets.items():
            zs = []
            for i in range(0, len(Xs), 256):
                xb = ((Xs[i:i + 256].float() - mu) / sd).unsqueeze(1).to(DEVICE)
                zs.append(model(xb)[0].cpu())
            out[s] = (torch.cat(zs).numpy(), ys.numpy())
    return out


def stage2(ztr, ytr, t_logits, seed):
    """the mlp_kd readout from stage2_readout.py, unchanged."""
    torch.manual_seed(seed)
    mu, sd = ztr.mean(0, keepdims=True), ztr.std(0, keepdims=True) + 1e-6
    X = torch.from_numpy((ztr - mu) / sd).float().to(DEVICE)
    y = torch.from_numpy(ytr).long().to(DEVICE)
    head = nn.Sequential(nn.Linear(X.shape[1], PROJ_DIM), nn.ReLU(),
                         nn.Linear(PROJ_DIM, N_CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    tl = t_logits.to(DEVICE)
    for _ in range(HEAD_EPOCHS):
        head.train()
        opt.zero_grad()
        o = head(X)
        (lossf(o, y) + kd_logit_loss(o, tl, HEAD_T)).backward()
        opt.step()
        sched.step()
    head.eval()

    def predict(z):
        with torch.no_grad():
            return head(torch.from_numpy((z - mu) / sd).float().to(DEVICE)).argmax(1).cpu().numpy()
    return predict


def fidelity(zs, zt, mode):
    """cosine fit against the constant-vector floor, and how well the student
    reproduces the teacher's pairwise structure, which also works when the two
    have different widths."""
    zt = centre(zt, mode, zt.mean(0, keepdim=True)).numpy()
    zs = centre(torch.from_numpy(zs), mode).numpy()
    out = {}
    if zs.shape[1] == zt.shape[1]:
        u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)  # noqa: E731
        base = float((u(zt) @ (u(zt).mean(0) / np.linalg.norm(u(zt).mean(0)))).mean())
        cos = float((u(zs) * u(zt)).sum(1).mean())
        out["fit_pct"] = round((cos - base) / (1 - base) * 100, 1)
    n = min(600, len(zs))
    idx = np.random.default_rng(0).choice(len(zs), n, replace=False)
    u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)      # noqa: E731
    iu = np.triu_indices(n, 1)
    a = (u(zs[idx]) @ u(zs[idx]).T)[iu]
    b = (u(zt[idx]) @ u(zt[idx]).T)[iu]
    out["rel_corr"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description="Swap the KD target, keep everything else.")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"protocol={PROTOCOL}  epochs={EPOCHS}  seeds={args.seeds}", flush=True)

    tf, _ = teacher_features()
    Xtr, ytr, ids = load_inputs("train")
    if list(tf["train"]["ids"]) != list(ids):
        raise RuntimeError("teacher feature ids do not match the log-mel cache ids")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)
    t_logits = tf["train"]["logits"]

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for name in args.methods:
        spec = METHODS[name]
        target = build_target(spec[0], tf)
        print(f"{name}: target {spec[0]} {tuple(target['train'].shape)} "
              f"centring={spec[1]} loss={spec[2]}", flush=True)
        for seed in args.seeds:
            t0 = time.time()
            out = run_one(spec, target, data, seed)
            predict = stage2(out["train"][0], out["train"][1], t_logits, seed)
            r = {"protocol": PROTOCOL, "method": name, "target": spec[0],
                 "centring": spec[1], "loss": spec[2], "seed": seed,
                 "seconds": round(time.time() - t0, 1)}
            for s in SPLITS:
                z, y = out[s]
                r.update(metrics(y, predict(z), s))
                r.update({f"{k}_{s}": v for k, v in
                          fidelity(z, target[s], spec[1]).items()})
            new.append(r)
            print(f"  seed {seed}: test UA {r['test_ua']:.4f}  "
                  f"fit {r.get('fit_pct_test', float('nan'))}  "
                  f"rel {r['rel_corr_test']}  ({r['seconds']:.0f}s)", flush=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(
                RUNS_CSV, index=False)

    df = pd.read_csv(RUNS_CSV)
    print("\n=== test UA, mean over seeds ===")
    print(df.groupby("method").agg(
        n=("seed", "size"), test_ua=("test_ua", "mean"), sd=("test_ua", "std"),
        fit_train=("fit_pct_train", "mean"), fit_test=("fit_pct_test", "mean"),
        rel_test=("rel_corr_test", "mean")).round(4).to_string())
    print(f"\n-> {RUNS_CSV}")


if __name__ == "__main__":
    main()
