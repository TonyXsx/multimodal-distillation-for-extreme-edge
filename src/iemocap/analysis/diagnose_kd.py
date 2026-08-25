"""
Why does KD lift val and never test? Looking at the representations instead.

Nine KD configs so far - two temperatures, three targets, centred and
uncentred, three seeds each - all show the same thing: val UA above the CE
baseline, test UA at or below it. Two rounds of hyperparameter changes didn't
move the direction, so stop tuning and look at what the representations
actually are.

Four questions, each with a number attached.

1. Does the teacher class structure survive unseen speakers? Separability and
   kNN transfer on train, val and test separately. If the teacher embedding is
   sharply class-structured on train and much less so on test, then what the
   student copies is partly train-specific and can't help on test.

2. Is the teacher embedding entangled with speaker? Leave-one-out kNN on
   speaker identity against the majority rate. IEMOCAP has two speakers per
   session and no overlap across splits, so any speaker structure the student
   copies definitely won't transfer.

3. Does KD actually pull the student embedding toward the teacher? Trains a CE
   student and a KD student here and compares their bottlenecks against the
   teacher on the same utterances. If the KD one is no more teacher-like then
   the loss isn't doing its job. If it is more teacher-like and still no better
   on test, then the teacher structure just isn't what this task needs at this
   capacity.

4. Which classes get conflated, and is it the same pair on both sides?
   Class-centroid cosine.

Every figure writes its numbers to csv too, a plot is never the only record.

    python src/iemocap/analysis/diagnose_kd.py
    python src/iemocap/analysis/diagnose_kd.py --no-student --embed pca
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
    class_centroid_similarity, cosine_separability, embed_2d, group_predictability,
    knn_transfer, plot_class_facets, plot_grouped_bars, plot_similarity_heatmap,
    silhouette,
)
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_FEATURES, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    BATCH_SIZE, CLASSES, DEVICE, DROPOUT, EPOCHS, LABEL_SMOOTH, LAM_FEATURE,
    LAM_LOGIT, LR, N_CLASSES, PROJ_DIM, SMALL_KW, WEIGHT_DECAY,
    DSResNetSE, kd_feature_loss, kd_logit_loss, load_inputs, load_teacher_signals,
    normalizer,
)
from iemocap.teacher.data import load_split  # noqa: E402

ADAPTED = IEMOCAP_FEATURES / "iemocap4__qwen2.5-omni-3b-bf16-LORA__adapter_ep3__audio-tr"
PROBE_DIR = IEMOCAP_DATA / "teacher_probe" / "bottleneck" / "adapted"
OUT = IEMOCAP_OUTPUTS / "analysis"
SPLITS = ("train", "val", "test")


def teacher_bottleneck(feature_key="audio_mean_l27"):
    """teacher 64-d bottleneck per split, through the saved probe.

    Test is included on purpose. This is about the teacher's representation,
    it isn't a signal being handed to the student.
    """
    ck = torch.load(PROBE_DIR / feature_key / "checkpoint.pt", weights_only=False)
    probe = Probe(ck["in_dim"], [ck["bottleneck"]], len(ck["classes"]), dropout=0.0)
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    out = {}
    with torch.no_grad():
        for s in SPLITS:
            r = torch.load(ADAPTED / f"{s}_features.pt", weights_only=False)
            X = (r["features"][feature_key].float() - ck["mu"]) / ck["sd"]
            _, z = probe(X, return_bottleneck=True)
            out[s] = (z.numpy(), r["labels"].numpy(), list(r["sample_ids"]))
    return out


def train_student(kind, data, epochs, seed=42):
    """train one student, return its bottleneck on every split."""
    Xtr, ytr, mu, sd, teach, evalsets = data
    lam_logit, lam_feat = (0.0, 0.0) if kind == "ce" else (LAM_LOGIT, LAM_FEATURE)
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
            if lam_logit:
                loss = loss + lam_logit * kd_logit_loss(logits, teach["logits"][idx].to(DEVICE), 8.0)
            if lam_feat:
                loss = loss + lam_feat * kd_feature_loss(z, teach["z_audio"][idx].to(DEVICE))
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
    ap = argparse.ArgumentParser(description="Diagnose why KD lifts val but not test.")
    ap.add_argument("--embed", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--no-student", action="store_true", help="teacher-side analysis only")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    speakers = {s: load_split(s)["speaker"].to_numpy() for s in SPLITS}
    reps = {}

    print("teacher bottleneck (audio_mean_l27) ...", flush=True)
    tb = teacher_bottleneck()
    for s in SPLITS:
        reps[("teacher", s)] = (tb[s][0], tb[s][1])

    if not args.no_student:
        Xtr, ytr, ids_tr = load_inputs("train")
        evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
        mu, sd = normalizer(Xtr)
        teach = load_teacher_signals(ids_tr)
        data = (Xtr, ytr, mu, sd, teach, evalsets)
        for kind, name in (("ce", "student_ce"), ("kd", "student_kd")):
            print(f"training {name} ({args.epochs} epochs) ...", flush=True)
            for s, (z, y) in train_student(kind, data, args.epochs).items():
                reps[(name, s)] = (z, y)

    # 1 and 2, separability and speaker entanglement
    sep_rows, spk_rows = [], []
    for (name, split), (Z, y) in reps.items():
        sep_rows.append({"representation": name, "split": split, "n": len(y),
                         **cosine_separability(Z, y), "silhouette": silhouette(Z, y)})
        spk_rows.append({"representation": name, "split": split,
                         **group_predictability(Z, speakers[split], k=args.k)})
    sep = pd.DataFrame(sep_rows).sort_values(["representation", "split"])
    spk = pd.DataFrame(spk_rows).sort_values(["representation", "split"])
    sep.to_csv(OUT / "separability.csv", index=False)
    spk.to_csv(OUT / "speaker_entanglement.csv", index=False)
    print("\n=== class separability (cosine gap = within - between) ===")
    print(sep.to_string(index=False))
    print("\n=== speaker entanglement (leave-one-out kNN on speaker id) ===")
    print(spk.to_string(index=False))

    # 3, transfer. fit on train, score val and test
    knn_rows = []
    for name in dict.fromkeys(n for n, _ in reps):
        Ztr, ytr_ = reps[(name, "train")]
        for split in ("val", "test"):
            Ze, ye = reps[(name, split)]
            knn_rows.append({"representation": name, "eval_split": split,
                             **knn_transfer(Ztr, ytr_, Ze, ye, k=args.k)})
    knn = pd.DataFrame(knn_rows)
    knn.to_csv(OUT / "knn_transfer.csv", index=False)
    print("\n=== kNN transfer (fit on train, no training) ===")
    print(knn.to_string(index=False))

    # 4, which classes get conflated
    for name in dict.fromkeys(n for n, _ in reps):
        Z, y = reps[(name, "test")]
        cs = class_centroid_similarity(Z, y, CLASSES)
        cs.to_csv(OUT / f"centroid_{name}.csv")
        plot_similarity_heatmap(
            cs, OUT / f"fig_centroid_{name}.png",
            title=f"Class-centroid cosine — {name} (test)",
            subtitle="High off-diagonal = the representation conflates that pair")

    # figures
    plot_grouped_bars(sep, "gap", "split", "representation", OUT / "fig_separability.png",
                      title="Class separability by split",
                      subtitle="Cosine gap (within-class minus between-class). Higher is better.",
                      ylabel="cosine gap")
    plot_grouped_bars(spk, "acc", "split", "representation", OUT / "fig_speaker.png",
                      title="How well SPEAKER can be read off the embedding",
                      subtitle="Leave-one-out k-NN on speaker identity; dashed line = majority rate",
                      ylabel="speaker k-NN accuracy",
                      ref_line=float(spk["majority_rate"].max()), ref_label="majority rate")
    plot_grouped_bars(knn, "ua", "eval_split", "representation", OUT / "fig_knn.png",
                      title="Portable class structure (k-NN fitted on train)",
                      subtitle="Unweighted accuracy on unseen speakers, no training",
                      ylabel="kNN UA", ref_line=0.25, ref_label="chance")

    for name in dict.fromkeys(n for n, _ in reps):
        for split in ("val", "test"):
            Z, y = reps[(name, split)]
            Z2 = embed_2d(Z, method=args.embed, seed=0)
            plot_class_facets(Z2, y, CLASSES, OUT / f"fig_embed_{name}_{split}.png",
                              title=f"{name} — {split} ({args.embed.upper()})",
                              subtitle="One panel per class; grey = all other utterances")

    print(f"\nCSVs and figures -> {OUT}")


if __name__ == "__main__":
    main()
