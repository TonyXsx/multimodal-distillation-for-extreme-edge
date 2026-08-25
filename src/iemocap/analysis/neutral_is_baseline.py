"""
Is neutral just the speaker's own voice?

Every split-shift measurement puts neutral last. Lowest train-test centroid
cosine in all six representations, and in the student it's also the noisiest by
a factor of three to five. The explanation I think is right is mechanical, not
statistical: neutral isn't an emotion the speaker performs, it's the absence of
one, so its class centroid is just whatever that speaker's ordinary voice
sounds like. Swap the speakers and the centroid moves.

That makes a prediction you can check without training anything. Within each
class, how well can the speaker be identified from the teacher features alone?
If neutral really is the speaker baseline then its utterances should carry
speaker identity more strongly than angry, happy or sad, because nothing is
painting over it.

Two controls: pool all 10 speakers so the split doesn't confound it, and
subsample every class to the size of the smallest so the kNN task is equally
hard for all four. Speaker ID gets easier with less data and neutral is the
biggest class.

    python src/iemocap/analysis/neutral_is_baseline.py
    python src/iemocap/analysis/neutral_is_baseline.py --repeats 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from sklearn.preprocessing import normalize  # noqa: E402

from common.repr_analysis import group_predictability  # noqa: E402
from iemocap.analysis.cluster_features import teacher_reps  # noqa: E402
from iemocap.paths import IEMOCAP_MANIFEST, IEMOCAP_OUTPUTS, find_adapted_features  # noqa: E402
from iemocap.student.kd_common import CLASSES  # noqa: E402

OUT = IEMOCAP_OUTPUTS / "analysis"
SPLITS = ("train", "val", "test")


def speaker_ids():
    """speaker index per utterance, in the order the feature files store them."""
    man = pd.read_csv(IEMOCAP_MANIFEST).set_index("turn_id")
    d = find_adapted_features()
    names, per_split = sorted(man.speaker.unique()), {}
    lut = {s: i for i, s in enumerate(names)}
    for s in SPLITS:
        r = torch.load(d / f"{s}_features.pt", weights_only=False)
        per_split[s] = np.array([lut[man.loc[i, "speaker"]] for i in r["sample_ids"]])
    return per_split, names


def main():
    ap = argparse.ArgumentParser(description="Does neutral carry the most speaker identity?")
    ap.add_argument("--repeats", type=int, default=10, help="subsampling repeats")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spk, names = speaker_ids()
    print(f"{len(names)} speakers: {', '.join(names)}", flush=True)

    reps = {}
    for key in ("audio_mean_l27", "last_token"):
        hi, lo = teacher_reps(key)
        reps[f"teacher {key} [2048]"] = hi
        reps[f"teacher {key} [64]"] = lo

    # speaker identifiability only says the speaker info is present inside a
    # class. it doesn't say that info moves the class centroid, which is what a
    # split shift measures - speaker cues can sit orthogonal to the centroid and
    # never disturb it. so the second measurement matches the claim better: how
    # far apart the ten per-speaker centroids of a class are. if they disagree
    # the class has no stable direction and its centroid is just a property of
    # whoever got sampled
    rows = []
    for name, per_split in reps.items():
        Z = np.concatenate([per_split[s][0] for s in SPLITS])
        y = np.concatenate([per_split[s][1] for s in SPLITS])
        sp = np.concatenate([spk[s] for s in SPLITS])

        n_min = min((y == c).sum() for c in range(len(CLASSES)))
        for ci, cname in enumerate(CLASSES):
            idx_all = np.where(y == ci)[0]
            accs = []
            for r in range(args.repeats):
                rng = np.random.default_rng(r)
                idx = rng.choice(idx_all, n_min, replace=False)
                accs.append(group_predictability(Z[idx], sp[idx], k=args.k)["acc"])
            cents = normalize(np.stack([
                normalize(Z[(y == ci) & (sp == s_i)]).mean(0)
                for s_i in range(len(names))]))
            sim = cents @ cents.T
            iu = np.triu_indices(len(names), k=1)
            rows.append({"representation": name, "class": cname, "n_used": int(n_min),
                         "speaker_knn_acc": round(float(np.mean(accs)), 4),
                         "sd": round(float(np.std(accs, ddof=1)), 4),
                         "chance": round(1.0 / len(names), 4),
                         "speaker_centroid_cos": round(float(sim[iu].mean()), 4)})
            print(f"  {name:32s} {cname:8s} speaker k-NN {rows[-1]['speaker_knn_acc']:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "neutral_baseline.csv", index=False)
    print("\n=== speaker identifiability within each emotion (chance = %.3f) ===" % (1 / len(names)))
    print(df.pivot(index="representation", columns="class",
                   values="speaker_knn_acc").reindex(columns=CLASSES).to_string())
    print("\n=== agreement between the 10 per-speaker centroids of each class ===")
    print("    (low = the class has no direction of its own; its centroid follows the speakers)")
    print(df.pivot(index="representation", columns="class",
                   values="speaker_centroid_cos").reindex(columns=CLASSES).to_string())
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
