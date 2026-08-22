"""
How much information is actually in the teacher's soft labels?

Logit-KD can only help if the teacher's distribution says something a smoothed
one-hot label does not. Three things decide that, and they are separable:

    nontarget_mass       1 - p_max. How much probability sits outside the
                         teacher's own answer at all.
    nontarget_structure  KL(q_nontarget || uniform) / ln(C-1), in [0, 1].
                         Whether that leftover mass points at specific
                         confusable classes or is spread evenly.
    dark_knowledge       the product. This is KL(q || matched label smoothing)
                         normalised by ln(C-1) -- i.e. exactly the part of the
                         soft target that label smoothing could NOT have
                         produced. Zero means logit-KD is label smoothing with
                         extra steps.

The product is what matters, and unlike raw entropy it is comparable across
datasets with different class counts.

Two teacher heads are compared where both exist, because the three tracks
differ in a way that was never controlled:

    FSC      logit-KD distilled the PROBE's logits (2048->64->31 head)
    MIntRec  logit-KD distilled Qwen's own generative logits (30 classes)
    IEMOCAP  logit-KD distilled Qwen's own generative logits (4 classes)

A LoRA-tuned generative head is trained to emit one token and saturates; a
probe trained for 50 epochs of CE need not saturate the same way. If IEMOCAP's
dark_knowledge is near zero while FSC's is not, then "too few classes" and
"wrong teacher head" are both live explanations -- and the IEMOCAP probe rows,
which hold the class count fixed at 4 and change only the head, separate them.

Everything is measured on TRAIN, the only split whose teacher signal a student
ever sees.

Outputs (outputs/analysis/):
    dark_knowledge.csv       one row per (teacher head, temperature)
    fig_dark_knowledge.png   the product, per teacher, across temperatures

Usage:
    python src/analysis/dark_knowledge.py
    python src/analysis/dark_knowledge.py --temps 1 2 4 8 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.probe import Probe  # noqa: E402
from common.repr_analysis import plot_grouped_bars  # noqa: E402

ROOT = _SRC.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "analysis"

FSC_TAG = "fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34"
IEMOCAP_TAG = "iemocap4__qwen2.5-omni-3b-bf16-LORA__adapter_ep3__audio-tr"
MINTREC_TAG = "mintrec2.0__qwen2.5-omni-3b-4bit-QLORA__tva_tr__adapter_ep3"


def raw_logits(feat_dir, split="train"):
    """Qwen's own generative logits, restricted to the class tokens."""
    r = torch.load(DATA / feat_dir / f"{split}_features.pt", weights_only=False)
    return r["features"]["logits"].float(), r["labels"].long()


def fsc_probe_logits():
    """Rebuild exactly the signal fsc/student/kd_common.py hands the student."""
    ck_dir = DATA / "teacher_probe" / FSC_TAG / "checkpoints"
    ck = torch.load(next(ck_dir.glob("B2_*.pt")), weights_only=False)
    probe = Probe(2048, ck["hidden_dims"], ck["n_classes"], dropout=ck["dropout"])
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    d = torch.load(DATA / "teacher_features" / FSC_TAG / "train_features.pt",
                   weights_only=False)
    X = d["features"][ck["feature_name"]].float()
    X = (X - ck["standardizer"]["mean"]) / ck["standardizer"]["std"]
    with torch.no_grad():
        return probe(X).float(), d["labels"].long()


def iemocap_probe_logits(key):
    """The 2048->64->4 probe whose bottleneck is already the Feature-KD target."""
    ck = torch.load(DATA / "iemocap" / "teacher_probe" / "bottleneck" / "adapted" / key
                    / "checkpoint.pt", weights_only=False)
    probe = Probe(ck["in_dim"], [ck["bottleneck"]], len(ck["classes"]), dropout=0.0)
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    d = torch.load(DATA / "iemocap" / "teacher_features" / IEMOCAP_TAG
                   / "train_features.pt", weights_only=False)
    X = (d["features"][key].float() - ck["mu"]) / ck["sd"]
    with torch.no_grad():
        return probe(X).float(), d["labels"].long()


def content(logits, y, T):
    """Soft-target information content at temperature T."""
    C = logits.shape[1]
    q = F.softmax(logits / T, dim=1)
    p_max, arg = q.max(1)

    # renormalised distribution over the C-1 classes the teacher did not pick
    nt = q.scatter(1, arg[:, None], 0.0)
    nt = nt / nt.sum(1, keepdim=True).clamp_min(1e-12)
    # KL(nt || uniform) = ln(C-1) - H(nt); divided through so it lands in [0, 1]
    H_nt = -(nt.clamp_min(1e-12).log() * nt).sum(1)
    structure = (np.log(C - 1) - H_nt) / np.log(C - 1)
    mass = 1.0 - p_max

    H = -(q.clamp_min(1e-12).log() * q).sum(1)
    return {
        "C": C, "n": len(y), "T": T,
        "teacher_train_acc": round(float((arg == y).float().mean()), 4),
        "p_max": round(float(p_max.mean()), 4),
        "norm_entropy": round(float((H / np.log(C)).mean()), 4),
        "nontarget_mass": round(float(mass.mean()), 4),
        "nontarget_structure": round(float(structure.mean()), 4),
        "dark_knowledge": round(float((mass * structure).mean()), 4),
        # same quantity un-normalised: comparable in absolute terms, and it exposes
        # the ln(C-1) ceiling that a 4-class teacher cannot get past
        "dark_knowledge_nats": round(float((mass * structure).mean() * np.log(C - 1)), 4),
        "ceiling_nats": round(float(np.log(C - 1)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Dark-knowledge content of each teacher's soft labels.")
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sources = {
        "FSC probe head [31]": fsc_probe_logits,
        "MIntRec Qwen head [30]": lambda: raw_logits("mintrec/teacher_features/" + MINTREC_TAG),
        "IEMOCAP Qwen head [4]": lambda: raw_logits("iemocap/teacher_features/" + IEMOCAP_TAG),
        "IEMOCAP probe audio [4]": lambda: iemocap_probe_logits("audio_mean_l27"),
        "IEMOCAP probe lasttoken [4]": lambda: iemocap_probe_logits("last_token"),
    }

    rows = []
    for name, fn in sources.items():
        logits, y = fn()
        print(f"{name}: {tuple(logits.shape)}", flush=True)
        for T in args.temps:
            rows.append({"teacher": name, **content(logits, y, T)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dark_knowledge.csv", index=False)

    bars = df.copy()
    bars["Tlabel"] = ["T=%g" % t for t in bars["T"]]
    plot_grouped_bars(bars, "dark_knowledge", "Tlabel", "teacher",
                      OUT / "fig_dark_knowledge.png",
                      title="How much of the soft label could label smoothing not have produced?",
                      subtitle="nontarget_mass x nontarget_structure on each train split. "
                               "0 means logit-KD is label smoothing with extra steps.",
                      ylabel="dark knowledge (normalised)")

    cols = ["teacher", "C", "teacher_train_acc", "p_max", "norm_entropy",
            "nontarget_mass", "nontarget_structure", "dark_knowledge",
            "dark_knowledge_nats", "ceiling_nats"]
    for T in args.temps:
        print("\n=== T = %g ===" % T)
        print(df[df["T"] == T][cols].to_string(index=False))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
