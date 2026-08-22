"""
What the 2048-d teacher features, their 64-d bottlenecks, and the student's own
embedding actually look like -- and what is lost at each step.

The distillation chain has three stages, and each one can destroy structure:

    Qwen hidden state [2048]  ->  probe bottleneck [64]  ->  student z [64]

Accuracy alone cannot say where the loss happens. Six representations are
embedded and scored side by side so it can be read off directly:

    teacher audio_mean_l27  at 2048 and at 64   (the clean, reachable feature)
    teacher last_token      at 2048 and at 64   (the privileged readout)
    student CE              at 64               (no teacher signal at all)
    student Feature-KD      at 64               (aligned to the audio bottleneck)

Everything is shown on TEST -- unseen speakers, the only split where the
numbers mean anything. Train is scored too but only in the metrics table, to
expose how much of the teacher's apparent structure is the probe memorising
its 3,205 training samples.

The 2048-d features are standardised with TRAIN statistics before embedding,
which is exactly what the probe consumes, so the 2048 and 64 panels differ by
the bottleneck alone rather than by preprocessing.

Outputs (outputs/iemocap/analysis/):
    fig_cluster_chain.png        six t-SNE panels, one per representation
    cluster_metrics.csv          separability, silhouette, k-NN transfer
    cluster_centroids.csv        class-centroid cosine per representation

Usage:
    python src/iemocap/analysis/cluster_features.py
    python src/iemocap/analysis/cluster_features.py --embed pca --epochs 20
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
from common.augment import spec_augment  # noqa: E402
from common.probe import Probe  # noqa: E402
from common.repr_analysis import (  # noqa: E402
    class_centroid_similarity, cosine_separability, embed_2d, knn_transfer,
    plot_embedding_grid, plot_similarity_heatmap, silhouette,
)
from iemocap.paths import IEMOCAP_OUTPUTS, IEMOCAP_PROBE, find_adapted_features  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, CLASSES, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LAM_FEATURE,
    LR, N_CLASSES, PROJ_DIM, SMALL_KW, WEIGHT_DECAY,
    DSResNetSE, kd_feature_loss, load_inputs, load_teacher_signals, normalizer,
)

OUT = IEMOCAP_OUTPUTS / "analysis"
SPLITS = ("train", "val", "test")
KEYS = {"audio_mean_l27": "audio_mean_l27", "last_token": "last_token"}


def teacher_reps(key):
    """Standardised 2048-d feature and its 64-d bottleneck, for every split."""
    ck = torch.load(IEMOCAP_PROBE / "bottleneck" / "adapted" / key / "checkpoint.pt",
                    weights_only=False)
    probe = Probe(ck["in_dim"], [ck["bottleneck"]], len(ck["classes"]), dropout=0.0)
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    d = find_adapted_features()
    hi, lo = {}, {}
    with torch.no_grad():
        for s in SPLITS:
            r = torch.load(d / f"{s}_features.pt", weights_only=False)
            X = (r["features"][key].float() - ck["mu"]) / ck["sd"]
            _, z = probe(X, return_bottleneck=True)
            hi[s] = (X.numpy(), r["labels"].numpy())
            lo[s] = (z.numpy(), r["labels"].numpy())
    return hi, lo


def train_student(kind, data, epochs, seed=42):
    """CE-only or Feature-KD (audio bottleneck target); returns z per split."""
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
            loss = ce(logits, ytr[idx].to(DEVICE))
            if kind == "featkd":
                loss = loss + LAM_FEATURE * kd_feature_loss(z, t_z[idx].to(DEVICE))
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


def main():
    ap = argparse.ArgumentParser(description="Cluster the distillation chain, stage by stage.")
    ap.add_argument("--embed", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    reps = {}
    for key in KEYS:
        print(f"teacher {key} ...", flush=True)
        hi, lo = teacher_reps(key)
        reps[f"teacher {key} [2048]"] = hi
        reps[f"teacher {key} [64]"] = lo

    Xtr, ytr, ids_tr = load_inputs("train")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    teach = load_teacher_signals(ids_tr)
    data = (Xtr, ytr, mu, sd, teach["z_audio"], evalsets)
    for kind, name in (("ce", "student CE [64]"), ("featkd", "student Feature-KD [64]")):
        print(f"training {name} ({args.epochs} epochs) ...", flush=True)
        reps[name] = train_student(kind, data, args.epochs)

    rows = []
    for name, per_split in reps.items():
        for split in SPLITS:
            Z, y = per_split[split]
            r = {"representation": name, "split": split, "dim": Z.shape[1], "n": len(y),
                 **cosine_separability(Z, y), "silhouette": silhouette(Z, y)}
            if split != "train":
                Ztr, ytr_ = per_split["train"]
                r.update({f"knn_{k}": v for k, v in
                          knn_transfer(Ztr, ytr_, Z, y, k=args.k).items()})
            rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "cluster_metrics.csv", index=False)

    cents = []
    for name, per_split in reps.items():
        Z, y = per_split["test"]
        cs = class_centroid_similarity(Z, y, CLASSES)
        cs.insert(0, "representation", name)
        cents.append(cs.reset_index().rename(columns={"index": "class"}))
    pd.concat(cents).to_csv(OUT / "cluster_centroids.csv", index=False)

    print(f"\nembedding with {args.embed.upper()} ...", flush=True)
    panels = []
    for name, per_split in reps.items():
        Z, y = per_split["test"]
        g = m[(m.representation == name) & (m.split == "test")].iloc[0]
        panels.append((name, embed_2d(Z, method=args.embed, seed=0), y,
                       f"gap {g['gap']:.3f}   sil {g['silhouette']:+.3f}   kNN UA {g['knn_ua']:.3f}"))
    plot_embedding_grid(panels, CLASSES, OUT / "fig_cluster_chain.png",
                        title="Where the class structure goes: 2048-d teacher feature "
                              "-> 64-d bottleneck -> student",
                        subtitle=f"IEMOCAP test split, 1,085 unseen-speaker utterances, "
                                 f"{args.embed.upper()} on cosine distance")

    for name, per_split in reps.items():
        Z, y = per_split["test"]
        safe = name.replace(" ", "_").replace("[", "").replace("]", "")
        plot_similarity_heatmap(class_centroid_similarity(Z, y, CLASSES),
                                OUT / f"fig_centroid_{safe}.png",
                                title=f"Class-centroid cosine — {name} (test)",
                                subtitle="High off-diagonal = the two classes are conflated")

    print("\n=== test split ===")
    print(m[m.split == "test"][["representation", "dim", "within", "between", "gap",
                                "silhouette", "knn_wa", "knn_ua"]].to_string(index=False))
    print("\n=== train (how much is memorised) ===")
    print(m[m.split == "train"][["representation", "gap", "silhouette"]].to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
