"""
Does LoRA-adapting the teacher help or hurt the student?

The adapted teacher memorises its training split - a twenty-parameter head on
four principal components of its features still fits that split at 0.906 - and
that memorisation is what closes its logit channel. The obvious question is
whether a frozen Qwen used as a plain audio encoder would distil better. It
would cost target quality: on this protocol the 64-d probe scores 0.786 test UA
on the adapted feature against 0.723 on the frozen one. This measures what that
trade is worth to the student.

Only the feature source changes. Both arms standardise on their own train split,
build the same two interfaces, and are read out with the same two heads; the
soft labels, where used, are the adapted head's in both arms, because the frozen
extraction has no head of its own and holding the logit channel fixed is what
isolates the feature.

SI protocol only - the frozen extraction was never run per LOSO fold. One
speaker-independent split, so this indicates a sign rather than settling it.

    IEMOCAP_PROTOCOL=si python src/iemocap/student/frozen_vs_adapted.py
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
from iemocap.paths import IEMOCAP_DATA, IEMOCAP_OUTPUTS, PROTOCOL  # noqa: E402
from iemocap.student.kd_common import (  # noqa: E402
    N_CLASSES, available_splits, load_inputs, normalizer,
)
from iemocap.student.logit_channel import readout, train_encoder  # noqa: E402
from iemocap.student.lowdim_target import knn_ua, r2  # noqa: E402
from iemocap.student.target_variants import (  # noqa: E402
    centre, metrics, probe_activations, train_probe,
)

OUT = IEMOCAP_OUTPUTS / "student"
RUNS_CSV = OUT / "frozen_vs_adapted.csv"
SPLITS = available_splits()
SEEDS = [42, 43, 44, 45, 46]
FEAT_KEY = "audio_mean_l27"
ARMS = [(arm, iface) for arm in ("adapted", "frozen") for iface in ("probe64", "pca64")]


def bank(arm):
    """one teacher extraction, standardised on its own train split."""
    root = IEMOCAP_DATA / "teacher_features"
    pat = "*LORA*" if arm == "adapted" else "*frozen*"
    fd = sorted(p for p in root.glob(pat) if p.is_dir())[-1]
    raw = {}
    for s in SPLITS:
        d = torch.load(fd / f"{s}_features.pt", weights_only=False, map_location="cpu")
        raw[s] = {"h": d["features"][FEAT_KEY].float(), "y": d["labels"].numpy(),
                  "ids": d["sample_ids"],
                  "logits": d["features"].get("logits", torch.empty(0)).float()}
    mu = raw["train"]["h"].mean(0, keepdim=True)
    sd = raw["train"]["h"].std(0, keepdim=True).clamp_min(1e-6)
    for s in SPLITS:
        raw[s]["h"] = (raw[s]["h"] - mu) / sd
    return raw


def build_target(iface, tf, ytr):
    if iface == "pca64":
        p = PCA(n_components=64, svd_solver="full", random_state=0).fit(tf["train"]["h"].numpy())
        return {s: torch.from_numpy(p.transform(tf[s]["h"].numpy())).float() for s in SPLITS}
    m = train_probe(tf["train"]["h"], ytr, 50)
    return {s: probe_activations(m, tf[s]["h"], False) for s in SPLITS}


def main():
    ap = argparse.ArgumentParser(description="Frozen teacher against the adapted one.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    if PROTOCOL != "si":
        raise SystemExit(f"the frozen extraction only exists for si, got {PROTOCOL}")
    OUT.mkdir(parents=True, exist_ok=True)

    banks = {a: bank(a) for a in ("adapted", "frozen")}
    Xtr, ytr, ids = load_inputs("train")
    ytr = ytr.numpy()
    for a, b in banks.items():
        if list(b["train"]["ids"]) != list(ids):
            raise RuntimeError(f"{a} feature ids do not match the log-mel cache ids")
    evalsets = {s: load_inputs(s)[:2] for s in SPLITS}
    mu, sd = normalizer(Xtr)
    data = (Xtr, ytr, mu, sd, evalsets)
    logits = banks["adapted"]["train"]["logits"]      # the same soft labels for both arms

    existing = pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame()
    new = []
    for arm, iface in ARMS:
        tf = banks[arm]
        target = build_target(iface, tf, ytr)
        t_mu = target["train"].mean(0, keepdim=True)
        knn = knn_ua(target["train"].numpy(), ytr, target["test"].numpy(), tf["test"]["y"])
        print(f"{arm}/{iface}: target k-NN {knn}", flush=True)
        for seed in args.seeds:
            t0 = time.time()
            out = train_encoder("pca", 64, target, logits, 2.0, data, seed)
            base = {"protocol": PROTOCOL, "arm": arm, "interface": iface, "seed": seed,
                    "target_knn": knn, "seconds": round(time.time() - t0, 1)}
            for s in SPLITS:
                base[f"r2_{s}"] = r2(centre(torch.from_numpy(out[s][0]), "perdim").numpy(),
                                     centre(target[s], "perdim", t_mu).numpy())
            line = []
            for rname, lg in (("ce", None), ("kd", logits)):
                predict = readout(out["train"][0], out["train"][1], lg, 2.0, seed)
                r = dict(base, readout=rname)
                for s in SPLITS:
                    r.update(metrics(out[s][1], predict(out[s][0]), s))
                new.append(r)
                line.append(f"{rname} {r['test_ua']:.4f}")
            print(f"  seed {seed}: " + "  ".join(line) + f"  ({base['seconds']:.0f}s)", flush=True)
            pd.concat([existing, pd.DataFrame(new)], ignore_index=True).to_csv(RUNS_CSV, index=False)

    df = pd.read_csv(RUNS_CSV)
    print("\n=== test UA, mean over seeds ===")
    print(df.pivot_table(index=["arm", "interface"], columns="readout",
                         values="test_ua", aggfunc="mean").round(4).to_string())
    print("\n=== target k-NN and how much of it the student reproduces ===")
    print(df.groupby(["arm", "interface"]).agg(
        knn=("target_knn", "first"), r2_train=("r2_train", "mean"),
        r2_test=("r2_test", "mean")).round(4).to_string())
    print(f"\n-> {RUNS_CSV}")


if __name__ == "__main__":
    main()
