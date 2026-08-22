"""
Why does joint Feature-KD do nothing while the same cosine loss, used alone in
stage 1, works?

Both optimise `1 - cos(z_student, z_teacher)` against the same target. The only
difference is whether CE is allowed to act on the encoder at the same time. The
accuracy column says one works and one does not; it cannot say why. Three things
can, and all three are computable from the cached embeddings without retraining
anything:

    target fidelity   cos(student z, teacher z) on train and on test. Splits
                      "never learned the mapping" from "learned it on train and
                      it did not carry". Read against the collapse baseline --
                      a student that emits one constant vector for everything
                      still scores well, because the target has a large shared
                      direction.
    emotion content   k-NN transfer train -> test. What the embedding is for.
    speaker content   leave-one-out k-NN predicting WHICH OF THE 10 SPEAKERS
                      produced the utterance, inside the test split, classes
                      balanced by subsampling. What the embedding leaked.

The speaker measurement is the reason this runs on SD and not SI. Under SI the
three splits are three disjoint sets of speakers, so "which split" and "which
speaker" are the same question and a high score cannot distinguish a speaker
code from a session or channel code. Under SD all ten speakers appear in every
split, so the two are independent and speaker identity can be measured directly.

Outputs (outputs/iemocap/analysis/):
    what_student_encodes.csv        per encoder x seed: fidelity, emotion, speaker
    fig_emotion_vs_speaker.png      the two contents side by side, per encoder
    fig_speaker_tsne.png            test embeddings coloured by speaker

Usage:
    IEMOCAP_PROTOCOL=sd python src/iemocap/analysis/what_the_student_encodes.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.probe import Probe  # noqa: E402
from common.repr_analysis import (  # noqa: E402
    cosine_separability, embed_2d, group_predictability, knn_transfer, plot_grouped_bars,
)
from iemocap.paths import (  # noqa: E402
    IEMOCAP_MANIFEST, IEMOCAP_OUTPUTS, IEMOCAP_PROBE, IEMOCAP_STUDENT,
    find_adapted_features,
)
from iemocap.student.kd_common import CLASSES  # noqa: E402

OUT = IEMOCAP_OUTPUTS / "analysis"
ZCACHE = IEMOCAP_STUDENT / "z_cache"
SPLITS = ("train", "val", "test")
TARGETS = ("audio_mean_l27", "last_token")
ORDER = ["ce", "logit_kd", "feature_kd", "feature_kd_audio",
         "full_kd_lasttoken", "full_kd_audio", "feature_only", "feature_only_audio"]


def teacher_z(key):
    """The 64-d probe bottleneck for this protocol, every split."""
    ck = torch.load(IEMOCAP_PROBE / "bottleneck" / "adapted" / key / "checkpoint.pt",
                    weights_only=False)
    probe = Probe(ck["in_dim"], [ck["bottleneck"]], len(ck["classes"]), dropout=0.0)
    probe.load_state_dict(ck["state_dict"])
    probe.eval()
    d = find_adapted_features()
    out = {}
    with torch.no_grad():
        for s in SPLITS:
            r = torch.load(d / f"{s}_features.pt", weights_only=False)
            X = (r["features"][key].float() - ck["mu"]) / ck["sd"]
            out[s] = (probe(X, return_bottleneck=True)[1].numpy(), r["labels"].numpy(),
                      list(r["sample_ids"]))
    return out


def collapse_floor(t):
    """Score a student that emits one constant vector for every utterance."""
    tn = normalize(t)
    c = tn.mean(0)
    return float((tn @ (c / np.linalg.norm(c))).mean())


def fidelity(zs, zt):
    a, b = normalize(zs), normalize(zt)
    return float((a * b).sum(1).mean())


def balanced_speaker_knn(Z, spk, k, repeats, rng0=0):
    """k-NN accuracy at naming the speaker, classes equalised by subsampling."""
    ids, counts = np.unique(spk, return_counts=True)
    n_min = counts.min()
    accs = []
    for r in range(repeats):
        rng = np.random.default_rng(rng0 + r)
        idx = np.concatenate([rng.choice(np.where(spk == s)[0], n_min, replace=False)
                              for s in ids])
        accs.append(group_predictability(Z[idx], spk[idx], k=k)["acc"])
    return float(np.mean(accs)), float(np.std(accs)), len(ids), int(n_min)


def main():
    ap = argparse.ArgumentParser(description="What the student's embedding actually encodes.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--embed", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tz = {key: teacher_z(key) for key in TARGETS}
    ref_ids = {s: tz[TARGETS[0]][s][2] for s in SPLITS}
    ref_y = {s: tz[TARGETS[0]][s][1] for s in SPLITS}
    floors = {key: {s: collapse_floor(tz[key][s][0]) for s in SPLITS} for key in TARGETS}

    man = pd.read_csv(IEMOCAP_MANIFEST).set_index("turn_id")
    names = sorted(man.speaker.unique())
    lut = {s: i for i, s in enumerate(names)}
    spk = {s: np.array([lut[man.loc[i, "speaker"]] for i in ref_ids[s]]) for s in SPLITS}
    print(f"{len(names)} speakers, chance = {1/len(names):.3f}; "
          f"test has {len(spk['test'])} utterances")

    files = sorted(ZCACHE.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no cached embeddings in {ZCACHE}")

    rows, panels = [], {}
    for f in files:
        enc = f.stem.split("_seed")[0]
        seed = int(f.stem.split("_seed")[1].split("_")[0])
        d = torch.load(f, weights_only=False)
        if not np.array_equal(d["train"]["y"], ref_y["train"]):
            raise RuntimeError(f"{f.name}: cached order does not match the manifest")

        ztr, ztest = d["train"]["z"], d["test"]["z"]
        r = {"encoder": enc, "seed": seed}
        for key in TARGETS:
            tag = "audio" if key.startswith("audio") else "lasttoken"
            for s, zs in (("train", ztr), ("test", ztest)):
                fid = fidelity(zs, tz[key][s][0])
                fl = floors[key][s]
                r[f"fid_{tag}_{s}"] = round(fid, 4)
                # share of the distance from "constant vector" to "exact match"
                r[f"fit_{tag}_{s}_pct"] = round((fid - fl) / (1 - fl) * 100, 1)
        r["emotion_knn_ua"] = round(knn_transfer(ztr, d["train"]["y"], ztest,
                                                 d["test"]["y"], k=args.k)["ua"], 4)
        r["emotion_gap_train"] = round(cosine_separability(ztr, d["train"]["y"])["gap"], 4)
        r["emotion_gap_test"] = round(cosine_separability(ztest, d["test"]["y"])["gap"], 4)
        acc, sd, n_spk, n_min = balanced_speaker_knn(ztest, spk["test"], args.k, args.repeats)
        r.update({"speaker_knn_acc": round(acc, 4), "speaker_knn_sd": round(sd, 4),
                  "speaker_chance": round(1 / n_spk, 4), "speaker_n_per_class": n_min})
        rows.append(r)
        panels.setdefault(enc, (ztest, d["test"]["y"], spk["test"]))
        print(f"  {enc:19s} seed {seed}  emotion {r['emotion_knn_ua']:.4f}  "
              f"speaker {acc:.4f}  fit(train/test) "
              f"{r['fit_lasttoken_train_pct']:.0f}%/{r['fit_lasttoken_test_pct']:.0f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "what_student_encodes.csv", index=False)

    agg = df.groupby("encoder").mean(numeric_only=True).reindex(ORDER)
    print("\n=== what each encoder's 64-d embedding contains (test, mean over seeds) ===")
    print(agg[["emotion_knn_ua", "speaker_knn_acc", "emotion_gap_train", "emotion_gap_test",
               "fit_lasttoken_train_pct", "fit_lasttoken_test_pct",
               "fit_audio_train_pct", "fit_audio_test_pct"]].round(4).to_string())

    bars = pd.concat([
        pd.DataFrame({"encoder": agg.index, "metric": "emotion (k-NN UA, chance .25)",
                      "v": agg.emotion_knn_ua.values}),
        pd.DataFrame({"encoder": agg.index, "metric": "speaker (k-NN acc, chance .10)",
                      "v": agg.speaker_knn_acc.values})])
    plot_grouped_bars(bars, "v", "encoder", "metric", OUT / "fig_emotion_vs_speaker.png",
                      title="What the student's 64-d embedding encodes",
                      subtitle="IEMOCAP SD test split, 5 seeds. Emotion is transferred from "
                               "train; speaker is leave-one-out within test, classes balanced.",
                      ylabel="k-NN accuracy")

    if not args.no_embed:
        import matplotlib.pyplot as plt
        from common.repr_analysis import INK, INK_SOFT, SURFACE, _style
        show = [e for e in ["ce", "feature_kd", "logit_kd", "feature_only"] if e in panels]
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c2367f",
                "#a8760a", "#00868b", "#8b2f5f", "#5b7c1f", "#9a4a1f"]
        fig, axes = plt.subplots(2, len(show), figsize=(3.5 * len(show), 7.2), facecolor=SURFACE)
        for j, enc in enumerate(show):
            Z, y, s = panels[enc]
            Z2 = embed_2d(Z, method=args.embed, seed=0)
            for row, (lab, groups, pal) in enumerate([
                    ("by emotion", y, cols[:len(CLASSES)]), ("by speaker", s, cols)]):
                ax = axes[row, j]
                for i in np.unique(groups):
                    m = groups == i
                    ax.scatter(Z2[m, 0], Z2[m, 1], s=6, c=pal[i % len(pal)], linewidths=0,
                               alpha=0.75)
                ax.set_xticklabels([]), ax.set_yticklabels([])
                _style(ax)
                if row == 0:
                    ax.set_title(enc, fontsize=11, color=INK)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=10, color=INK_SOFT)
        fig.suptitle("Same embeddings, coloured two ways", fontsize=13, color=INK, x=0.02,
                     ha="left")
        fig.text(0.02, 0.955, "IEMOCAP SD test split, seed 42. Emotion structure on top, "
                              "speaker structure below.", fontsize=9, color=INK_SOFT)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUT / "fig_speaker_tsne.png", dpi=160, facecolor=SURFACE,
                    bbox_inches="tight")
        plt.close(fig)

    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
