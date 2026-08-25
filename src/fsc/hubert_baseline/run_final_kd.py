"""
Final test run for the HuBERT baseline. README in this folder has the why.

Close mirror of fsc/student/final_test.py: same small student, same KD
hyperparameters (T=8, lam_logit=1.0, lam_feature=1.0), same four methods, same
protocol of picking on val and touching test once. The only thing that changes
is which teacher produced ztr/ltr.

SMALL_KW, T_KD, LAM_LOGIT, LAM_FEATURE, METHODS and load_test() are imported
from final_test.py rather than retyped, so the two runs can't drift apart on
model size or KD config.

Outputs go somewhere separate, nothing here overwrites the Qwen run.
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
from common.config import DATA_ROOT, OUTPUTS_ROOT                   # noqa: E402
from common.models.audio_student import DSResNetSE, model_summary   # noqa: E402
from fsc.student.kd_common import (                                   # noqa: E402
    evaluate, spec_augment, kd_logit_loss, kd_feature_loss,
    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, LABEL_SMOOTH, SEED, DEVICE,
)
from fsc.student.final_test import (                                  # noqa: E402
    T_KD, LAM_LOGIT, LAM_FEATURE, SMALL_KW, METHODS, load_test,
)
from fsc.hubert_baseline.kd_common_hubert import load_data            # noqa: E402

CKPT_DIR = DATA_ROOT / "student" / "hubert_final_test_checkpoints"
OUT = OUTPUTS_ROOT / "fsc" / "hubert_baseline" / "final_test"
QWEN_RESULTS_CSV = OUTPUTS_ROOT / "fsc" / "student" / "final_test" / "results.csv"
for d in (CKPT_DIR, OUT):
    d.mkdir(parents=True, exist_ok=True)


def run(method, train_data, val_data, Xte, yte):
    ll, lf = METHODS[method]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Xtr, ytr, ztr, ltr = train_data
    Xva, yva, _, _ = val_data

    model = DSResNetSE(**SMALL_KW).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    n = Xtr.shape[0]
    best_f1, best_state, best_val, best_epoch = -1.0, None, None, -1
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = spec_augment(Xtr[idx].to(DEVICE))
            yb = ytr[idx].to(DEVICE)
            opt.zero_grad()
            z_s, logits_s = model(xb)
            loss = ce(logits_s, yb)
            if ll > 0:
                loss = loss + ll * kd_logit_loss(logits_s, ltr[idx].to(DEVICE), T_KD)
            if lf > 0:
                loss = loss + lf * kd_feature_loss(z_s, ztr[idx].to(DEVICE).float())
            loss.backward()
            opt.step()
        sched.step()

        m_val = evaluate(model, Xva, yva)               # selection only
        if m_val["macro_f1"] > best_f1:
            best_f1 = m_val["macro_f1"]
            best_val = m_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # the one test pass
    model.load_state_dict(best_state)
    m_test = evaluate(model, Xte, yte)

    torch.save({"method": method, "state_dict": best_state, "best_epoch": best_epoch,
                "val_metrics": best_val, "test_metrics": m_test,
                "T": T_KD, "lam_logit": ll, "lam_feature": lf,
                "teacher": "hubert-large-ll60k (frozen, audio-only)"},
               CKPT_DIR / f"{method}.pt")
    return best_val, m_test, best_epoch


def main():
    print(f"Device: {DEVICE}")
    sz = model_summary(DSResNetSE(**SMALL_KW))
    print(f"Small student params: {sz['params']:,}  | FP32 {sz['fp32_mb']:.2f} MB  | INT8(est) {sz['int8_mb']:.2f} MB")
    print(f"Teacher: HuBERT-large-ll60k (frozen, audio-only)  |  "
          f"KD: T={T_KD:g}, lam_logit={LAM_LOGIT:g}, lam_feature={LAM_FEATURE:g}  | seed={SEED}\n")

    train_data, val_data = load_data()
    Xte, yte = load_test()
    print(f"test {tuple(Xte.shape)}\n")

    results = []
    for method in METHODS:
        print(f"=== {method} ===")
        val_m, test_m, best_epoch = run(method, train_data, val_data, Xte, yte)
        print(f"  selected @epoch {best_epoch+1} (val macroF1={val_m['macro_f1']:.4f})  ->  "
              f"TEST acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} wF1={test_m['weighted_f1']:.4f}")
        results.append({"method": method,
                        "test_acc": round(test_m["acc"], 4),
                        "test_macro_f1": round(test_m["macro_f1"], 4),
                        "test_weighted_f1": round(test_m["weighted_f1"], 4),
                        "val_macro_f1_sel": round(val_m["macro_f1"], 4),
                        "best_epoch": best_epoch + 1,
                        "params": sz["params"], "fp32_mb": round(sz["fp32_mb"], 2),
                        "int8_mb": round(sz["int8_mb"], 2)})

    csv_path = OUT / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "test_acc", "test_macro_f1", "test_weighted_f1",
                                          "val_macro_f1_sel", "best_epoch", "params", "fp32_mb", "int8_mb"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")

    ce = results[0]
    print(f"\n{'method':<12}{'TEST acc':<10}{'TEST macroF1':<14}{'gain vs CE (macroF1)'}")
    print("-" * 56)
    for r in results:
        print(f"{r['method']:<12}{r['test_acc']:<10.4f}{r['test_macro_f1']:<14.4f}"
              f"{r['test_macro_f1'] - ce['test_macro_f1']:+.4f}")

    make_comparison(results)
    print("Done.")


def make_comparison(hubert_results):
    """overlay the Qwen final_test results if they're there. read-only."""
    qwen_results = None
    if QWEN_RESULTS_CSV.exists():
        with open(QWEN_RESULTS_CSV, newline="", encoding="utf-8") as f:
            qwen_results = list(csv.DictReader(f))

    if qwen_results:
        comp_path = OUT / "comparison.csv"
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "qwen_test_acc", "qwen_test_macro_f1",
                        "hubert_test_acc", "hubert_test_macro_f1"])
            for m in METHODS:
                q = next((r for r in qwen_results if r["method"] == m), None)
                h = next(r for r in hubert_results if r["method"] == m)
                w.writerow([m, q["test_acc"] if q else "", q["test_macro_f1"] if q else "",
                            h["test_acc"], h["test_macro_f1"]])
        print(f"Comparison CSV -> {comp_path}")

    methods = list(METHODS.keys())
    h_f1 = [next(r["test_macro_f1"] for r in hubert_results if r["method"] == m) for m in methods]
    q_f1 = [float(next(r["test_macro_f1"] for r in qwen_results if r["method"] == m)) for m in methods] if qwen_results else None

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w / 2, h_f1, w, label="HuBERT-large teacher (audio-only, frozen)", color="#d62728")
    if q_f1:
        ax.bar(x + w / 2, q_f1, w, label="Qwen2.5-Omni teacher (multimodal, prompted)", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Test Macro F1")
    vals_all = h_f1 + (q_f1 or [])
    ax.set_ylim(min(vals_all) - 0.02, max(vals_all) + 0.02)
    ax.set_title("Audio-only (HuBERT) vs multimodal (Qwen2.5-Omni) teacher\n"
                 "same small DSResNet-SE student, same KD recipe (T=8, lam_logit=1.0, lam_feature=1.0)",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.4f", fontsize=8, rotation=90, padding=2)
    fig.savefig(OUT / "final_test_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {OUT / 'final_test_comparison.png'}")


if __name__ == "__main__":
    main()
