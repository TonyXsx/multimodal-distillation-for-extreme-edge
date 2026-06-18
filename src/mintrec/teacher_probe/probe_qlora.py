"""
Linear/MLP probe on the QLoRA-adapted teacher features (MIntRec2.0, 30-class intent).

Answers two things after the one QLoRA fine-tune:
  1. How strong is the adapted teacher's readout?  -> argmax of the saved `logits`
     (this is the teacher itself, no probe needed; should match training dev ~0.63).
  2. Did the adapted hidden features become a better KD target than the FROZEN ones,
     and did the CLEAN audio_mean ride along?  -> probe audio_mean_l27 / audio_mean_final
     / last_token, train on train, eval on dev, side-by-side with the frozen baseline.

Strict protocol (same as the frozen probe): standardize with TRAIN stats, fixed
50-epoch AdamW(1e-3, wd 1e-4), batch 256, CE, no dev selection.

Outputs: outputs/mintrec/teacher_probe/qlora_probe.csv  (+ printed table)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA, MINTREC_OUTPUTS  # noqa: E402
from common.probe import Probe                           # noqa: E402

EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE, SEED = 50, 1e-3, 1e-4, 256, 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = [("A1", [], "linear"), ("A2", [1024], "1024-MLP")]

QLORA_DIR = MINTREC_DATA / "teacher_features" / "mintrec2.0__qwen2.5-omni-3b-4bit-QLORA__tva_tr__adapter_ep3"
FROZEN_DIR = MINTREC_DATA / "teacher_features" / "mintrec2.0__qwen2.5-omni-3b-4bit__pf_text-video4f-audio__plain__audiomean"
OUT_DIR = MINTREC_OUTPUTS / "teacher_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def standardize(Xtr, Xev):
    m = Xtr.mean(0, keepdim=True); s = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return (Xtr - m) / s, (Xev - m) / s


def train_one(hidden, Xtr, ytr, Xev, yev, n_classes):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Probe(Xtr.shape[1], hidden, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    Xtr, ytr, Xev, yev = Xtr.to(DEVICE), ytr.to(DEVICE), Xev.to(DEVICE), yev.to(DEVICE)
    n = len(Xtr)
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad(); loss_fn(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
        pred = model(Xev).argmax(1)
    return tr_acc, (pred == yev).float().mean().item(), f1_score(yev.cpu().numpy(), pred.cpu().numpy(), average="macro")


def probe_feature(name, Xtr, ytr, Xev, yev, nc, results, src):
    Xtr_n, Xev_n = standardize(Xtr.float(), Xev.float())
    for aid, hid, desc in ARCHS:
        tra, da, df = train_one(hid, Xtr_n, ytr, Xev_n, yev, nc)
        results.append({"source": src, "feature": name, "arch": aid, "arch_desc": desc,
                        "dev_acc": round(da, 4), "dev_macro_f1": round(df, 4), "train_acc": round(tra, 4)})
        print(f"  {src:8} {name:18} [{aid}] dev_acc={da:.4f} macroF1={df:.4f} (train={tra:.3f})")


def main():
    assert (QLORA_DIR / "train_features.pt").exists() and (QLORA_DIR / "dev_features.pt").exists(), \
        f"QLoRA features not found in {QLORA_DIR} (run extract_with_lora.py --split all first)"
    tr = torch.load(QLORA_DIR / "train_features.pt", weights_only=False)
    dv = torch.load(QLORA_DIR / "dev_features.pt", weights_only=False)
    ytr, yev = tr["labels"].long(), dv["labels"].long()
    nc = int(max(ytr.max(), yev.max())) + 1
    print(f"Device {DEVICE} | train {len(ytr)} dev {len(yev)} | classes {nc}\n")

    results = []
    # (0) teacher's OWN readout: argmax of saved logits (no probe) — should match training dev
    dev_logits_acc = (dv["features"]["logits"].float().argmax(1) == yev).float().mean().item()
    dev_logits_f1 = f1_score(yev.numpy(), dv["features"]["logits"].float().argmax(1).numpy(), average="macro")
    print(f"[teacher readout] dev_acc={dev_logits_acc:.4f} macroF1={dev_logits_f1:.4f}  (argmax of logits, no probe)\n")
    results.append({"source": "QLoRA", "feature": "logits(argmax)", "arch": "-", "arch_desc": "teacher readout",
                    "dev_acc": round(dev_logits_acc, 4), "dev_macro_f1": round(dev_logits_f1, 4), "train_acc": ""})

    # (1) probe the adapted features (KD-target candidates)
    print("ADAPTED (QLoRA) features:")
    for key in ("audio_mean_l27", "audio_mean_final", "last_token"):
        if key in tr["features"]:
            probe_feature(key, tr["features"][key], ytr, dv["features"][key], yev, nc, results, "QLoRA")

    # (2) frozen baseline for side-by-side (note: different input order/pooling — indicative, not identical pipeline)
    if (FROZEN_DIR / "train_features.pt").exists():
        ftr = torch.load(FROZEN_DIR / "train_features.pt", weights_only=False)
        fdv = torch.load(FROZEN_DIR / "dev_features.pt", weights_only=False)
        fytr, fyev = ftr["labels"].long(), fdv["labels"].long()
        print("\nFROZEN baseline (audio_mean):")
        for key in ("pf_audio_mean_l27", "pf_audio_mean_L24-27-30-34"):
            if key in ftr["features"]:
                probe_feature(key, ftr["features"][key], fytr, fdv["features"][key], fyev, nc, results, "frozen")
    else:
        print("\n(frozen baseline folder not found — skipping side-by-side)")

    csv_path = OUT_DIR / "qlora_probe.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "feature", "arch", "arch_desc",
                                          "dev_acc", "dev_macro_f1", "train_acc"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults CSV -> {csv_path}")


if __name__ == "__main__":
    main()
