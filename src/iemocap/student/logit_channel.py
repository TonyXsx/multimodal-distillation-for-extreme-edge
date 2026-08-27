"""
The two channels, crossed: what the student is taught in stage 1 against what
supplies the soft labels in stage 2.

The five-fold decomposition says the feature channel carries most of the gain
(hubert +3.96 pp with no logit anywhere, of +4.92 total) and that qwen's feature
channel is the one that fails (+0.86). The logit diagnostics say qwen's soft
labels are not noise - the non-target ordering is more structured than hubert's
(eta^2 0.275 vs 0.212, confusion alignment 0.79 vs 0.60) - only squashed, and
T = 4.4 is where its non-target mass matches hubert's at T = 2.

Both are testable by crossing the two. The encoder dominates the cost and the
readouts take two seconds each, so every encoder is scored under every readout.

    encoders    pca4 pca64          the unsupervised interface, two widths
                probe64             the label-trained bottleneck, as control
                celogit_T2/T44      single stage, CE + logit-KD, the (a) arm
    readouts    ce                  no soft labels at all
                qwen_T2 qwen_T44    the same logits, squashed and unsquashed
                pcahead             logits from a short-trained head on pca4
                hubert_T2           the other teacher's logits on our features

Fold 1, five seeds, everything else the fixed protocol.

    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/logit_channel.py
    IEMOCAP_PROTOCOL=loso1 IEMOCAP_TEACHER=hubert python \
        src/iemocap/student/logit_channel.py --encoders pca64 probe64
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.losses import kd_feature_loss, kd_logit_loss  # noqa: E402
from common.models.audio_student import DSResNetSE  # noqa: E402
from iemocap.analysis.logit_channel_diagnostics import (  # noqa: E402
    match_temperature, nontarget, soft,
)
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_OUTPUTS, PROTOCOL, TEACHER  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LR, N_CLASSES, PROJ_DIM,
    SMALL_KW, WEIGHT_DECAY, available_splits, load_inputs, normalizer,
)
from iemocap.student.lowdim_target import r2  # noqa: E402
from iemocap.student.target_variants import (  # noqa: E402
    HEAD_EPOCHS, HEAD_LR, centre, metrics, train_probe, probe_activations,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "logit_channel.csv"
HEADS_CSV = OUT / "logit_channel_heads.csv"
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]
FEAT_KEYS = {"qwen": "audio_mean_l27", "hubert": "hubert_mean_l18"}
PCA_HEAD_EPOCHS = 5      # short on purpose: a head that cannot memorise its split

# name -> (recipe, width). celogit arms are single stage and ignore the width.
ENCODERS = {"pca4": ("pca", 4), "pca64": ("pca", 64), "probe64": ("probe", 64),
            "celogit_T2": ("celogit", 64), "celogit_T44": ("celogit", 64)}


def bank(teacher):
    """feature, labels and head logits for one teacher, both splits."""
    tsuf = "" if teacher == "qwen" else f"_{teacher}"
    root = IEMOCAP_DATA / f"teacher_features{tsuf}_{PROTOCOL}"
    fd = sorted(p for p in root.glob("*LORA*") if p.is_dir())[-1]
    pdir = (IEMOCAP_DATA / f"teacher_probe{tsuf}_{PROTOCOL}" / "bottleneck" /
            "adapted" / FEAT_KEYS[teacher])
    ck = torch.load(pdir / "checkpoint.pt", weights_only=False, map_location="cpu")
    out = {}
    for s in SPLITS:
        d = torch.load(fd / f"{s}_features.pt", weights_only=False, map_location="cpu")
        h = d["features"][FEAT_KEYS[teacher]].float()
        out[s] = {"h": (h - ck["mu"]) / ck["sd"], "y": d["labels"].numpy(),
                  "ids": d["sample_ids"], "logits": d["features"]["logits"].float()}
    return out


def pca_head_logits(tf, ytr, k=4, epochs=PCA_HEAD_EPOCHS):
    """a deliberately weak readout of the teacher: a linear head on k PCA
    components, stopped early. With k=4 it has 20 parameters, so unlike the
    LoRA head it cannot memorise the split it supplies targets for."""
    p = PCA(n_components=k, svd_solver="full", random_state=0).fit(tf["train"]["h"].numpy())
    Z = {s: torch.from_numpy(p.transform(tf[s]["h"].numpy())).float() for s in SPLITS}
    torch.manual_seed(0)
    head = nn.Linear(k, N_CLASSES)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    y = torch.from_numpy(ytr).long()
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            F.cross_entropy(head(Z["train"][idx]), y[idx]).backward()
            opt.step()
    with torch.no_grad():
        return {s: head(Z[s]).detach() for s in SPLITS}


def head_report(name, logits, y, ref_mass):
    """the saturation diagnostic, for whatever supplies the soft labels."""
    lg = logits.numpy() if torch.is_tensor(logits) else logits
    return {"protocol": PROTOCOL, "head": name,
            "train_acc": round(float((lg.argmax(1) == y).mean()), 4),
            "nontarget_mass_T2": round(float(nontarget(soft(lg, 2.0), y)[0].mean()), 4),
            "T_to_match_hubert": round(float(match_temperature(lg, y, ref_mass)), 3)}


def build_target(recipe, k, tf, ytr):
    if recipe == "pca":
        p = PCA(n_components=k, svd_solver="full", random_state=0).fit(tf["train"]["h"].numpy())
        return {s: torch.from_numpy(p.transform(tf[s]["h"].numpy())).float() for s in SPLITS}
    m = train_probe(tf["train"]["h"], ytr, 50)
    return {s: probe_activations(m, tf[s]["h"], False) for s in SPLITS}


def train_encoder(recipe, dim, target, t_logits, kd_t, data, seed):
    """feature-only stage 1, or the single-stage CE + logit-KD arm."""
    Xtr, ytr, mu, sd, evalsets = data
    t_z = None if target is None else centre(
        target["train"], "perdim", target["train"].mean(0, keepdim=True)).to(DEVICE)
    yb = torch.from_numpy(ytr).long().to(DEVICE)
    tl = t_logits.to(DEVICE)
    lossf = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=dim,
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
            z, o = model(xb)
            if recipe == "celogit":
                loss = lossf(o, yb[idx]) + kd_logit_loss(o, tl[idx], kd_t)
            else:
                loss = kd_feature_loss(centre(z, "perdim"), t_z[idx])
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


def readout(ztr, ytr, t_logits, kd_t, seed):
    """stage2_readout.py's mlp_kd head, with the soft-label term optional."""
    torch.manual_seed(seed)
    mu, sd = ztr.mean(0, keepdims=True), ztr.std(0, keepdims=True) + 1e-6
    X = torch.from_numpy((ztr - mu) / sd).float().to(DEVICE)
    y = torch.from_numpy(ytr).long().to(DEVICE)
    head = nn.Sequential(nn.Linear(X.shape[1], PROJ_DIM), nn.ReLU(),
                         nn.Linear(PROJ_DIM, N_CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    tl = None if t_logits is None else t_logits.to(DEVICE)
    for _ in range(HEAD_EPOCHS):
        head.train()
        opt.zero_grad()
        o = head(X)
        loss = lossf(o, y) if tl is None else lossf(o, y) + kd_logit_loss(o, tl, kd_t)
        loss.backward()
        opt.step()
        sched.step()
    head.eval()

    def predict(z):
        with torch.no_grad():
            return head(torch.from_numpy((z - mu) / sd).float().to(DEVICE)).argmax(1).cpu().numpy()
    return predict


def main():
    ap = argparse.ArgumentParser(description="Cross the feature and logit channels.")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tf = bank(TEACHER)
    other = bank("hubert" if TEACHER == "qwen" else "qwen")
    Xtr, ytr, ids = load_inputs("train")
    ytr = ytr.numpy()          # every head below indexes with it
    if list(tf["train"]["ids"]) != list(ids) or list(other["train"]["ids"]) != list(ids):
        raise RuntimeError("teacher feature ids do not match the log-mel cache ids")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)

    own = tf["train"]["logits"]
    hub = (other if TEACHER == "qwen" else tf)["train"]["logits"]
    ref = float(nontarget(soft(hub.numpy(), 2.0), ytr)[0].mean())
    t44 = match_temperature(own.numpy(), ytr, ref)
    pca_lg = pca_head_logits(tf, ytr)["train"]

    reports = [head_report(n, lg, ytr, ref) for n, lg in
               (("own_lora", own), ("hubert_lora", hub), ("pca4_head", pca_lg))]
    pd.DataFrame(reports).to_csv(HEADS_CSV, index=False)
    print(f"teacher={TEACHER}  matched T={t44:.2f}  (hubert non-target mass {ref:.4f})")
    print(pd.DataFrame(reports).to_string(index=False), f"\n-> {HEADS_CSV}\n", flush=True)

    # name -> (logits, temperature)
    READOUTS = {"ce": (None, 0.0), "qwen_T2": (own, 2.0), "qwen_T44": (own, t44),
                "pcahead": (pca_lg, 2.0), "hubert_T2": (hub, 2.0)}

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for name in args.encoders:
        recipe, k = ENCODERS[name]
        kd_t = t44 if name.endswith("T44") else 2.0
        target = None if recipe == "celogit" else build_target(recipe, k, tf, ytr)
        print(f"{name}: {recipe} dim={k} kd_t={kd_t:.2f}", flush=True)
        for seed in args.seeds:
            t0 = time.time()
            out = train_encoder(recipe, k, target, own, kd_t, data, seed)
            base = {"protocol": PROTOCOL, "teacher": TEACHER, "encoder": name,
                    "recipe": recipe, "dim": k, "seed": seed,
                    "seconds": round(time.time() - t0, 1)}
            if target is not None:
                t_mu = target["train"].mean(0, keepdim=True)
                for s in SPLITS:
                    base[f"r2_{s}"] = r2(centre(torch.from_numpy(out[s][0]), "perdim").numpy(),
                                         centre(target[s], "perdim", t_mu).numpy())
            line = []
            for rname, (lg, T) in READOUTS.items():
                predict = readout(out["train"][0], out["train"][1], lg, T, seed)
                r = dict(base, readout=rname, kd_t=round(T, 3))
                for s in SPLITS:
                    r.update(metrics(out[s][1], predict(out[s][0]), s))
                new.append(r)
                line.append(f"{rname} {r['test_ua']:.4f}")
            print(f"  seed {seed}: " + "  ".join(line) + f"  ({base['seconds']:.0f}s)", flush=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(
                RUNS_CSV, index=False)

    df = pd.read_csv(RUNS_CSV)
    print("\n=== test UA, mean over seeds ===")
    print(df.pivot_table(index=["teacher", "encoder"], columns="readout",
                         values="test_ua", aggfunc="mean").round(4).to_string())
    print(f"\n-> {RUNS_CSV}")


if __name__ == "__main__":
    main()
