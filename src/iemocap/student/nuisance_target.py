"""
Three ways to attack the generalisation gap, on the target rather than the student.

reach_teacher_space.py ruled out capacity, the loss and reachability: the
student fits the 64-d target at R2 0.73 on train and 0.01 on test. lowdim_target.py
then showed that shrinking the target to four components takes test R2 to 0.51,
so a large part of what does not transfer is nuisance the student is being told
to reproduce. Under LOSO the nuisance with a name is the speaker.

    noaug       stage 1 without SpecAugment, so the student sees the same audio
                the teacher's target was computed on. Beyer et al. (2022) make
                consistent views the first thing to get right; ours are not.
    spkproj     the between-speaker directions are projected out of the 2048-d
                feature before the PCA. On train this is exactly per-speaker
                centring, but as a fixed projection it applies to test too, so
                nothing about the test speakers is needed at inference.
    dropspk     axis-aligned control: of the leading 2k components, keep the k
                whose variance is least explained by speaker identity. The eta^2
                table says this should not work - speaker sits at 0.16 at most
                and is spread over every component, while the one component most
                worth keeping is also mildly speaker-related - so it is here to
                show the difference between dropping a direction and removing it.

The selection uses speaker identity only. Emotion eta^2 is reported alongside
for diagnosis and is never used to choose anything, which is what separates this
from the label-trained bottleneck the diagnostics blamed in the first place.

One fold, five seeds, everything else the fixed protocol.

    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/nuisance_target.py
    IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/nuisance_target.py --variants noaug
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.augment import spec_augment  # noqa: E402
from common.losses import kd_feature_loss  # noqa: E402
from common.models.audio_student import DSResNetSE  # noqa: E402
from iemocap.paths import PROTOCOL, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, LR, N_CLASSES, PROJ_DIM, SMALL_KW,
    WEIGHT_DECAY, available_splits, load_inputs, normalizer,
)
from iemocap.student.lowdim_target import knn_ua, pca_bank, probe_ua, r2  # noqa: E402
from iemocap.student.target_variants import (  # noqa: E402
    centre, metrics, stage2, teacher_features,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "nuisance_target.csv"
ETA_CSV = OUT / "nuisance_component_eta2.csv"
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]
TOP_M = 64          # how many components the eta^2 report covers
N_SPK_DIRS = 7      # 8 train speakers, so the between-speaker scatter has rank 7

# name -> (target recipe, width, SpecAugment in stage 1)
VARIANTS = {
    "base":      ("pca", 64, True),
    "noaug":     ("pca", 64, False),
    "noaug4":    ("pca", 4, False),
    "spkproj64": ("spkproj", 64, True),
    "spkproj8":  ("spkproj", 8, True),
    "spkproj4":  ("spkproj", 4, True),
    "dropspk8":  ("dropspk", 8, True),
}


def speakers(ids):
    """Ses02F_impro01_F000 -> Ses02_F. the suffix is the speaker; the prefix is
    only whoever led the scenario."""
    return np.array([f"{t.split('_')[0][:5]}_{t.rsplit('_', 1)[1][0]}" for t in ids])


def eta2(z, g):
    """one-way ANOVA per column: the share of a column's variance that group
    membership explains. 1 means the column is the group and nothing else."""
    g = np.asarray(g)
    mu, out = z.mean(0), np.zeros(z.shape[1])
    for u in np.unique(g):
        m = g == u
        out += m.sum() * (z[m].mean(0) - mu) ** 2
    return out / len(z) / np.maximum(z.var(0), 1e-12)


def speaker_dirs(X, spk, r=N_SPK_DIRS):
    """the leading eigenvectors of the between-speaker scatter."""
    mu = X.mean(0)
    B = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for u in np.unique(spk):
        d = (X[spk == u].mean(0) - mu)[:, None]
        B += (spk == u).sum() * d @ d.T
    w, v = np.linalg.eigh(B)
    return v[:, -r:].astype(np.float32)


def build_target(recipe, k, tf, spk_tr):
    """returns {split: tensor}, plus whatever is worth printing. Every statistic
    is fitted on train only."""
    H = {s: tf[s]["h"].numpy() for s in SPLITS}
    info = {}
    if recipe == "spkproj":
        V = speaker_dirs(H["train"], spk_tr)
        H = {s: H[s] - (H[s] @ V) @ V.T for s in SPLITS}
        info["spk_var_removed"] = round(float(
            1 - H["train"].var(0).sum() / tf["train"]["h"].numpy().var(0).sum()), 4)
    p = PCA(svd_solver="full", random_state=0).fit(H["train"])
    Z = {s: (H[s] - p.mean_) @ p.components_.T for s in SPLITS}
    cols = np.arange(k)
    if recipe == "dropspk":
        e = eta2(Z["train"][:, :2 * k], spk_tr)
        cols = np.sort(np.argsort(e)[:k])
        info["spk_eta2_kept"] = round(float(e[cols].mean()), 4)
        info["spk_eta2_dropped"] = round(float(np.delete(e, cols).mean()), 4)
    info["var_kept"] = round(float(p.explained_variance_ratio_[cols].sum()), 4)
    return {s: torch.from_numpy(Z[s][:, cols]).float() for s in SPLITS}, info


def run_one(target, dim, augment, data, seed, epochs):
    """lowdim_target.run_one with SpecAugment and the schedule made optional."""
    Xtr, ytr, mu, sd, evalsets = data
    t_z = centre(target["train"], "perdim", target["train"].mean(0, keepdim=True)).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DSResNetSE(n_mels=Xtr.shape[2], n_classes=N_CLASSES, proj_dim=dim,
                       dropout=DROPOUT, **SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            xb = ((Xtr[idx].float() - mu) / sd).unsqueeze(1).to(DEVICE)
            z, _ = model(spec_augment(xb) if augment else xb)
            opt.zero_grad()
            kd_feature_loss(centre(z, "perdim"), t_z[idx]).backward()
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


def component_report(tf, spk_tr, ytr):
    """which PCA components carry the speaker, and which carry the emotion."""
    p, _ = pca_bank(tf["train"]["h"])
    Z = (tf["train"]["h"].numpy() - p.mean_) @ p.components_[:TOP_M].T
    df = pd.DataFrame({"component": np.arange(TOP_M),
                       "var_ratio": p.explained_variance_ratio_[:TOP_M].round(5),
                       "spk_eta2": eta2(Z, spk_tr).round(4),
                       "emo_eta2": eta2(Z, ytr).round(4)})
    df.insert(0, "protocol", PROTOCOL)
    df.to_csv(ETA_CSV, index=False)
    o = df.sort_values("spk_eta2", ascending=False)
    print(f"speaker eta^2 over the first {TOP_M} components: "
          f"mean {df.spk_eta2.mean():.3f}  max {df.spk_eta2.max():.3f}")
    print("  most speaker-related:", list(o.component.iloc[:8]))
    print("  least speaker-related:", list(o.component.iloc[-8:][::-1]))
    print(f"  emotion eta^2 on those two sets: {o.emo_eta2.iloc[:8].mean():.3f} vs "
          f"{o.emo_eta2.iloc[-8:].mean():.3f}")
    print(f"-> {ETA_CSV}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Take the speaker out of the target.")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    # Beyer et al. pair consistent views with very long schedules; 70 is short
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tf, _ = teacher_features()
    Xtr, ytr, ids = load_inputs("train")
    if list(tf["train"]["ids"]) != list(ids):
        raise RuntimeError("teacher feature ids do not match the log-mel cache ids")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)
    t_logits = tf["train"]["logits"]

    spk = {s: speakers(tf[s]["ids"]) for s in SPLITS}
    print(f"protocol={PROTOCOL}  seeds={args.seeds}  "
          + "  ".join(f"{s} {len(np.unique(spk[s]))} speakers" for s in SPLITS), flush=True)
    component_report(tf, spk["train"], ytr)

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for name in args.variants:
        recipe, k, augment = VARIANTS[name]
        target, info = build_target(recipe, k, tf, spk["train"])
        t_mu = target["train"].mean(0, keepdim=True)
        ceil = {"knn_ua": knn_ua(target["train"].numpy(), ytr,
                                 target["test"].numpy(), tf["test"]["y"]),
                "probe_ua": probe_ua(target["train"].numpy(), ytr,
                                     target["test"].numpy(), tf["test"]["y"])}
        print(f"{name}: {recipe} k={k} aug={augment} {info}  "
              f"target k-NN {ceil['knn_ua']}  linear {ceil['probe_ua']}", flush=True)

        for seed in args.seeds:
            t0 = time.time()
            out = run_one(target, k, augment, data, seed, args.epochs)
            predict = stage2(out["train"][0], out["train"][1], t_logits, seed)
            r = {"protocol": PROTOCOL, "variant": name, "recipe": recipe, "dim": k,
                 "augment": augment, "epochs": args.epochs, "seed": seed, **info, **ceil,
                 "seconds": round(time.time() - t0, 1)}
            for s in SPLITS:
                z, y = out[s]
                r.update(metrics(y, predict(z), s))
                r[f"r2_{s}"] = r2(centre(torch.from_numpy(z), "perdim").numpy(),
                                  centre(target[s], "perdim", t_mu).numpy())
            new.append(r)
            print(f"  seed {seed}: test UA {r['test_ua']:.4f}  "
                  f"r2 {r['r2_train']:.3f}/{r['r2_test']:.3f}  ({r['seconds']:.0f}s)",
                  flush=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(
                RUNS_CSV, index=False)

    df = pd.read_csv(RUNS_CSV)
    # rows written before --epochs existed ran the protocol default
    df["epochs"] = df.get("epochs", EPOCHS)
    df["epochs"] = df["epochs"].fillna(EPOCHS).astype(int)
    print("\n=== test UA, mean over seeds ===")
    print(df.groupby(["variant", "epochs"]).agg(
        n=("seed", "size"), dim=("dim", "first"), aug=("augment", "first"),
        knn=("knn_ua", "first"), test_ua=("test_ua", "mean"), sd=("test_ua", "std"),
        r2_train=("r2_train", "mean"), r2_test=("r2_test", "mean"),
    ).sort_values("test_ua", ascending=False).round(4).to_string())
    print(f"\n-> {RUNS_CSV}")


if __name__ == "__main__":
    main()
