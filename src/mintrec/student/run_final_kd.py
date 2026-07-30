"""
MIntRec2.0 tiny-student KD comparison: audio-only vs audio-visual, wrapping up
the MIntRec track (see src/mintrec/student/README.md for the full rationale).

No hyperparameter search here -- everything not specific to this comparison
is reused UNCHANGED from the FSC final recipe: T=8, lam_logit=1.0,
lam_feature=1.0, AdamW(lr=1e-3, wd=1e-4), cosine schedule, label smoothing
0.1, SpecAugment on the audio branch, 70 epochs, seed 42, best-by-dev-macroF1
checkpoint selection, ONE final pass on the held-out TEST split per method.

10 conditions total:

  Audio-only student (97,926 params, ~0.37 MiB FP32):
    ao_ce_only
    ao_logit_kd                 (vs teacher `logits`, the real QLoRA head)
    ao_feature_kd_audiohidden   (vs bottleneck(audio_mean_l27) -- CLEAN audio feature)
    ao_feature_kd_lasttoken     (vs bottleneck(last_token)     -- privileged, all-modality)
    ao_full_kd_audiohidden      (logit + feature_audiohidden)
    ao_full_kd_lasttoken        (logit + feature_lasttoken)

  Audio-visual student (115,471 params, ~0.44 MiB FP32):
    av_ce_only
    av_logit_kd
    av_feature_kd_lasttoken     (fusion z vs bottleneck(last_token) only -- see README
                                 for why audiohidden is skipped for this student)
    av_full_kd_lasttoken

Outputs:
  data/mintrec/student/final_test_checkpoints/<method>.pt
  outputs/mintrec/student/final_test/results.csv
  outputs/mintrec/student/final_test/final_test.png
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA, MINTREC_OUTPUTS            # noqa: E402
from common.augment import spec_augment                            # noqa: E402
from mintrec.student.models import AudioOnlyStudent, AudioVisualStudent, count_params  # noqa: E402
from mintrec.student.kd_common import (                            # noqa: E402
    load_data, load_test, evaluate, kd_logit_loss, kd_feature_loss,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTH, SEED, DEVICE,
    T_KD, LAM_LOGIT, LAM_FEATURE,
)

CKPT_DIR = MINTREC_DATA / "student" / "final_test_checkpoints"
OUT = MINTREC_OUTPUTS / "student" / "final_test"
for d in (CKPT_DIR, OUT):
    d.mkdir(parents=True, exist_ok=True)

# (name, model_ctor, is_audio_visual, lam_logit, lam_feature, feature_key)
CONDITIONS = [
    ("ao_ce_only",                AudioOnlyStudent,    False, 0.0,       0.0,         None),
    ("ao_logit_kd",               AudioOnlyStudent,    False, LAM_LOGIT, 0.0,         None),
    ("ao_feature_kd_audiohidden", AudioOnlyStudent,    False, 0.0,       LAM_FEATURE, "z_audiohidden"),
    ("ao_feature_kd_lasttoken",   AudioOnlyStudent,    False, 0.0,       LAM_FEATURE, "z_lasttoken"),
    ("ao_full_kd_audiohidden",    AudioOnlyStudent,    False, LAM_LOGIT, LAM_FEATURE, "z_audiohidden"),
    ("ao_full_kd_lasttoken",      AudioOnlyStudent,    False, LAM_LOGIT, LAM_FEATURE, "z_lasttoken"),
    ("av_ce_only",                AudioVisualStudent, True,  0.0,       0.0,         None),
    ("av_logit_kd",               AudioVisualStudent, True,  LAM_LOGIT, 0.0,         None),
    ("av_feature_kd_lasttoken",   AudioVisualStudent, True,  0.0,       LAM_FEATURE, "z_lasttoken"),
    ("av_full_kd_lasttoken",      AudioVisualStudent, True,  LAM_LOGIT, LAM_FEATURE, "z_lasttoken"),
]


def run(name, ctor, is_av, lam_logit, lam_feature, feat_key, train_data, dev_data, Xte, Fte, yte):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logmel, frames, labels = train_data["logmel"], train_data["frames"], train_data["labels"]
    logits_t = train_data["logits"]
    feat_t = train_data[feat_key] if feat_key else None
    Xdv, Fdv, ydv = dev_data["logmel"], dev_data["frames"], dev_data["labels"]

    model = ctor().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    n = logmel.shape[0]
    best_f1, best_state, best_dev, best_epoch = -1.0, None, None, -1
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = spec_augment(logmel[idx].to(DEVICE))
            fb = frames[idx].to(DEVICE) if is_av else None
            yb = labels[idx].to(DEVICE)
            opt.zero_grad()
            z_s, logits_s = model(xb, fb)
            loss = ce(logits_s, yb)
            if lam_logit > 0:
                loss = loss + lam_logit * kd_logit_loss(logits_s, logits_t[idx].to(DEVICE), T_KD)
            if lam_feature > 0:
                loss = loss + lam_feature * kd_feature_loss(z_s, feat_t[idx].to(DEVICE))
            loss.backward()
            opt.step()
        sched.step()

        m_dev = evaluate_av(model, Xdv, Fdv, ydv, is_av)
        if m_dev["macro_f1"] > best_f1:
            best_f1 = m_dev["macro_f1"]
            best_dev = m_dev
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    m_test = evaluate_av(model, Xte, Fte, yte, is_av)

    sz = count_params(model)
    torch.save({"method": name, "state_dict": best_state, "best_epoch": best_epoch,
                "dev_metrics": best_dev, "test_metrics": m_test,
                "lam_logit": lam_logit, "lam_feature": lam_feature, "feature_key": feat_key,
                "params": sz}, CKPT_DIR / f"{name}.pt")
    return best_dev, m_test, best_epoch, sz


@torch.no_grad()
def evaluate_av(model, X, F, y, is_av, batch_size=256):
    """Like common.training.evaluate but forwards the optional visual frames too."""
    model.eval()
    preds = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(DEVICE)
        fb = F[i:i + batch_size].to(DEVICE) if is_av else None
        _, logits = model(xb, fb)
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    from sklearn.metrics import accuracy_score, f1_score
    yt = y.numpy()
    return {"acc": accuracy_score(yt, preds), "macro_f1": f1_score(yt, preds, average="macro"),
            "weighted_f1": f1_score(yt, preds, average="weighted")}


def main():
    print(f"Device: {DEVICE}")
    print(f"KD: T={T_KD:g}, lam_logit={LAM_LOGIT:g}, lam_feature={LAM_FEATURE:g}  | "
          f"epochs={EPOCHS}  batch={BATCH_SIZE}  seed={SEED}\n")

    train_data, dev_data = load_data()
    Xte, Fte, yte = load_test()
    print(f"test logmel {tuple(Xte.shape)}  frames {tuple(Fte.shape)}\n")

    results = []
    for name, ctor, is_av, ll, lf, fk in CONDITIONS:
        print(f"=== {name} ===")
        dev_m, test_m, best_epoch, sz = run(name, ctor, is_av, ll, lf, fk,
                                            train_data, dev_data, Xte, Fte, yte)
        print(f"  selected @epoch {best_epoch+1} (dev macroF1={dev_m['macro_f1']:.4f})  ->  "
              f"TEST acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} wF1={test_m['weighted_f1']:.4f}")
        results.append({
            "method": name, "student": "audio_visual" if is_av else "audio_only",
            "lam_logit": ll, "lam_feature": lf, "feature_key": fk or "-",
            "test_acc": round(test_m["acc"], 4), "test_macro_f1": round(test_m["macro_f1"], 4),
            "test_weighted_f1": round(test_m["weighted_f1"], 4),
            "dev_macro_f1_sel": round(dev_m["macro_f1"], 4), "best_epoch": best_epoch + 1,
            "params": sz, "fp32_mib": round(sz * 4 / 1024 ** 2, 3),
        })

    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "student", "lam_logit", "lam_feature", "feature_key",
                                          "test_acc", "test_macro_f1", "test_weighted_f1",
                                          "dev_macro_f1_sel", "best_epoch", "params", "fp32_mib"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")

    print(f"\n{'method':<28}{'TEST acc':<10}{'TEST macroF1':<14}")
    print("-" * 52)
    for r in results:
        print(f"{r['method']:<28}{r['test_acc']:<10.4f}{r['test_macro_f1']:<14.4f}")

    make_plot(results)
    print("Done.")


def make_plot(results):
    methods = [r["method"] for r in results]
    f1s = [r["test_macro_f1"] for r in results]
    colors = ["#7f7f7f" if "ce_only" in m else ("#1f77b4" if m.startswith("ao_") else "#2ca02c") for m in methods]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(methods))
    bars = ax.bar(x, f1s, color=colors)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test Macro F1 (MIntRec2.0, 30 classes)")
    ax.set_ylim(0, max(f1s) + 0.08)
    ax.set_title("MIntRec2.0 tiny-student KD wrap-up: audio-only (blue) vs audio-visual (green)\n"
                 "gray = CE-only baseline per student  ·  same KD recipe as FSC (T=8, lam=1.0/1.0)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "final_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {OUT / 'final_test.png'}")


if __name__ == "__main__":
    main()
