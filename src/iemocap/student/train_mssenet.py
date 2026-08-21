"""
Phase 1: MS-SENet student, CE baseline only -- no distillation yet.

The point of swapping the backbone is not the architecture; it is to run the
later CE / logit-KD / feature-KD / combined-KD comparison on a student that is
not itself the bottleneck. So this stage answers one question first: on OUR
protocol, what does a faithful MS-SENet reach with cross-entropy alone,
against the 96K DSResNet-SE's test UA of roughly 0.56-0.58?

WHAT IS KEPT FROM THE OFFICIAL IMPLEMENTATION
    architecture      exact port (src/common/models/mssenet.py)
    features          39-dim MFCC, 22050 Hz, 310000 samples, hop 512
    optimiser         Adam(lr=1e-3, betas=(0.93, 0.98), eps=1e-8)
    batch size        64
    label smoothing   0.1
    epochs            200 (official default; --epochs to shorten)

WHAT IS DELIBERATELY NOT KEPT -- the evaluation protocol
    The official code runs `KFold(n_splits=10, shuffle=True)` over UTTERANCES.
    IEMOCAP has ten speakers, so shuffled utterance folds put the same speaker
    in train and test; its reported numbers are not speaker-independent. It
    also passes the test fold in as `validation_data`.

    This script keeps our protocol unchanged: train = Sessions 2-4, val =
    Session 5, test = Session 1, all speaker-disjoint; selection on validation
    UA; test evaluated once on the selected checkpoint. That makes MS-SENet
    directly comparable to the DSResNet-SE numbers already in
    outputs/iemocap/student/, and it means results here should NOT be compared
    with the paper's published IEMOCAP figures, which were produced under the
    looser scheme.

`metrics` and the evaluation loop are imported from train_student.py rather
than reimplemented, so the two backbones are scored by identical code.

Outputs:
    outputs/iemocap/student/mssenet_ce{tag}.csv    one row per seed + summary

Usage:
    python src/iemocap/student/train_mssenet.py --seeds 42 --epochs 20   # quick look
    python src/iemocap/student/train_mssenet.py                          # 3 seeds, 200 epochs
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.models.mssenet import MSSENet  # noqa: E402
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.student.kd_common import CLASSES, DEVICE, LABEL_SMOOTH, N_CLASSES  # noqa: E402
from iemocap.student.train_student import metrics  # noqa: E402  identical scoring

MFCC_DIR = IEMOCAP_DATA / "student" / "mfcc39"
OUT_DIR = IEMOCAP_OUTPUTS / "student"

# Official MS-SENet settings.
LR, BETAS, EPS = 1e-3, (0.93, 0.98), 1e-8
BATCH_SIZE = 64
EPOCHS = 200


def load_split_mfcc(split):
    d = torch.load(MFCC_DIR / f"{split}.pt", weights_only=False)
    return d["X"], d["labels"]


@torch.no_grad()
def evaluate(model, X, y, batch=64):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        preds.append(model(X[i:i + batch].float().to(DEVICE))[1].argmax(1).cpu())
    return metrics(y.numpy(), torch.cat(preds).numpy())


def run(seed, data, epochs, deterministic=True):
    Xtr, ytr, Xva, yva, Xte, yte = data
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MSSENet(n_feat=Xtr.shape[2], n_classes=N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_ua, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:
                continue
            _, logits = model(Xtr[idx].float().to(DEVICE))
            loss = ce(logits, ytr[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = evaluate(model, Xva, yva)
        if m["ua"] > best_ua:
            best_ua, best_ep = m["ua"], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    fin_test = evaluate(model, Xte, yte)
    model.load_state_dict(best_state)
    val_m, test_m = evaluate(model, Xva, yva), evaluate(model, Xte, yte)
    return {"seed": seed, "best_epoch": best_ep,
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
            **{f"final_test_{k}": v for k, v in fin_test.items()}}


def main():
    ap = argparse.ArgumentParser(description="MS-SENet CE baseline on the IEMOCAP protocol.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--non-deterministic", action="store_true")
    args = ap.parse_args()

    if not (MFCC_DIR / "train.pt").exists():
        raise SystemExit(f"{MFCC_DIR} missing -- run precompute_mfcc.py first")
    Xtr, ytr = load_split_mfcc("train")
    Xva, yva = load_split_mfcc("val")
    Xte, yte = load_split_mfcc("test")
    data = (Xtr, ytr, Xva, yva, Xte, yte)

    n = sum(p.numel() for p in MSSENet(n_feat=Xtr.shape[2], n_classes=N_CLASSES).parameters())
    print(f"MS-SENet {n:,} params  {n * 4 / 1024**2:.2f} MiB fp32  {n / 1024**2:.2f} MiB int8")
    print(f"classes {CLASSES} | train {len(Xtr)} val {len(Xva)} test {len(Xte)} "
          f"| input {tuple(Xtr.shape[1:])} | epochs {args.epochs}\n")

    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / f"mssenet_ce{args.tag}.csv"
    for seed in args.seeds:
        t0 = time.time()
        r = run(seed, data, args.epochs, deterministic=not args.non_deterministic)
        r["seconds"] = round(time.time() - t0, 1)
        r["params"] = n
        rows.append(r)
        print(f"  seed={seed}  ep={r['best_epoch']:3d}  val UA={r['val_ua']:.4f}  "
              f"test UA={r['test_ua']:.4f}  test WA={r['test_wa']:.4f}  "
              f"macroF1={r['test_macro_f1']:.4f}  (final {r['final_test_ua']:.4f})  "
              f"[{r['seconds']:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(csv, index=False)

    df = pd.DataFrame(rows)
    print("\n=== MS-SENet CE baseline (mean over seeds) ===")
    for k in ("val_ua", "test_wa", "test_ua", "test_macro_f1", "final_test_ua"):
        print(f"  {k:18s} {df[k].mean():.4f} +/- {df[k].std():.4f}")
    print("\nDSResNet-SE CE reference (96K, same protocol): test UA ~0.56-0.58")
    print(f"CSV -> {csv}")


if __name__ == "__main__":
    main()
