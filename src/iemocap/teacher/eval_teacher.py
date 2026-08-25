"""
Evaluates the adapted IEMOCAP teacher on val and on held-out test.

The checkpoint is chosen on val UA during training, and this evaluates that
checkpoint once on test. It never scans epochs for the best test score. By
default it reads best_epoch out of the run config.json instead of taking it as
an argument, so the choice cannot be quietly changed after seeing test numbers.

Per split, and separately for the improvised and scripted slices:

    WA           weighted accuracy, i.e. plain accuracy
    UA           unweighted accuracy, macro recall. the selection metric
    macro F1
    weighted F1

The impro/scripted slice is why is_impro is in the manifest at all. The teacher
reads transcripts, and scripted IEMOCAP dialogues reuse fixed lines whose
wording correlates with the emotion. If the teacher is much better on the
scripted slice then some of what looks like emotion recognition is recitation,
and whatever the teacher knows for the wrong reason is what the student copies.

Per-utterance predictions get written too, so any other slice (speaker,
duration, class) can be recomputed without another GPU pass.

Everything lands under outputs/iemocap/teacher_lora/.

    python src/iemocap/teacher/eval_teacher.py \
        --run-dir data/iemocap/teacher_qlora/3b_bf16_audio-tr_r32a64
    python src/iemocap/teacher/eval_teacher.py --run-dir ... --epoch 2 --splits val
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, recall_score,
)
from tqdm import tqdm

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.teacher.backbone import load_thinker, get_hidden_size  # noqa: E402
from iemocap.teacher.data import (  # noqa: E402
    CLASSES, load_split, wav_path, load_audio, build_inputs, input_order_str,
)
from mintrec.teacher_probe.qlora_finetune import OmniClassifier  # noqa: E402


@torch.no_grad()
def predict_split(clf, proc, device, df, use_transcript, split):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"predict {split}", leave=False):
        try:
            wav = load_audio(wav_path(row))
            if wav.size == 0:
                raise ValueError("empty audio")
            inp = build_inputs(proc, device, wav, row["transcript"], use_transcript)
            logits = clf(inp)[0].float().cpu()
        except Exception as ex:
            print(f"  skip {row['turn_id']}: {type(ex).__name__}: {ex}")
            continue
        rows.append({
            "turn_id": row["turn_id"], "split": split, "speaker": row["speaker"],
            "is_impro": bool(row["is_impro"]), "duration": float(row["duration"]),
            "true": int(row["label"]), "pred": int(logits.argmax()),
            **{f"logit_{c}": round(float(logits[i]), 4) for i, c in enumerate(CLASSES)},
        })
    return pd.DataFrame(rows)


def score(sub):
    y, p = sub["true"].to_numpy(), sub["pred"].to_numpy()
    labels = list(range(len(CLASSES)))
    return {
        "n": len(sub),
        "wa": round(float(accuracy_score(y, p)), 4),
        "ua": round(float(recall_score(y, p, average="macro", labels=labels, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=labels, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, p, average="weighted", labels=labels, zero_division=0)), 4),
    }


def per_class(sub):
    labels = list(range(len(CLASSES)))
    pr, rc, f1, sup = precision_recall_fscore_support(
        sub["true"], sub["pred"], labels=labels, zero_division=0)
    return [{"class": CLASSES[i], "precision": round(float(pr[i]), 4),
             "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4),
             "support": int(sup[i])} for i in labels]


SLICES = [("all", lambda d: d),
          ("improvised", lambda d: d[d["is_impro"]]),
          ("scripted", lambda d: d[~d["is_impro"]])]


def main():
    ap = argparse.ArgumentParser(description="Evaluate the IEMOCAP LoRA teacher on val and test.")
    ap.add_argument("--run-dir", required=True, help="teacher_qlora/<tag> directory")
    ap.add_argument("--epoch", type=int, default=None,
                    help="override the val-selected epoch (use only for diagnostics)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N per split (smoke)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    epoch = args.epoch if args.epoch is not None else cfg["best_epoch"]
    if args.epoch is not None:
        print(f"WARNING: epoch overridden to {epoch} (val-selected was {cfg['best_epoch']})")
    adapter_dir = run_dir / f"adapter_ep{epoch}"
    head_path = run_dir / f"head_ep{epoch}.pt"
    dtype, use_transcript = cfg.get("dtype", "bf16"), bool(cfg["use_transcript"])
    tag = f"{run_dir.name}_ep{epoch}"

    print(f"run      : {run_dir.name}")
    print(f"epoch    : {epoch} (selected on {cfg.get('selection_metric', 'val_ua')})")
    print(f"dtype    : {dtype} | transcript: {use_transcript}")
    print(f"order    : {input_order_str(use_transcript)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_thinker(cfg["model_name"], dtype)
    hidden = get_hidden_size(model)
    from peft import PeftModel
    thinker = PeftModel.from_pretrained(model.thinker, str(adapter_dir)).eval()
    clf = OmniClassifier(thinker, hidden, len(CLASSES), pool="last",
                         head_hidden=cfg.get("head_hidden", 256)).to(device)
    clf.head.load_state_dict(torch.load(head_path, map_location=device))
    clf.head.to(torch.bfloat16)
    clf.eval()

    preds = pd.concat([predict_split(clf, proc, device, load_split(s, limit=args.limit),
                                     use_transcript, s) for s in args.splits],
                      ignore_index=True)

    summary, pc, conf = [], [], []
    for split in args.splits:
        d = preds[preds["split"] == split]
        for sname, fn in SLICES:
            sub = fn(d)
            if len(sub) == 0:
                continue
            summary.append({"run": run_dir.name, "epoch": epoch, "split": split,
                            "slice": sname, **score(sub)})
            for r in per_class(sub):
                pc.append({"split": split, "slice": sname, **r})
            if sname == "all":
                cm = confusion_matrix(sub["true"], sub["pred"], labels=list(range(len(CLASSES))))
                for i, t in enumerate(CLASSES):
                    for j, p in enumerate(CLASSES):
                        conf.append({"split": split, "true": t, "pred": p, "count": int(cm[i, j])})

    out = IEMOCAP_OUTPUTS / "teacher_lora"
    out.mkdir(parents=True, exist_ok=True)
    sm = pd.DataFrame(summary)
    sm.to_csv(out / f"eval_{tag}_summary.csv", index=False)
    pd.DataFrame(pc).to_csv(out / f"eval_{tag}_per_class.csv", index=False)
    pd.DataFrame(conf).to_csv(out / f"eval_{tag}_confusion.csv", index=False)
    preds.to_csv(out / f"eval_{tag}_predictions.csv", index=False)

    print("\n" + sm.to_string(index=False))
    print("\nper-class (all):")
    print(pd.DataFrame([r for r in pc if r["slice"] == "all"]).to_string(index=False))
    for split in args.splits:
        c = pd.DataFrame([r for r in conf if r["split"] == split])
        if len(c):
            print(f"\nconfusion [{split}] rows=true, cols=pred:")
            print(c.pivot(index="true", columns="pred", values="count")
                   .reindex(index=CLASSES, columns=CLASSES).to_string())
    print(f"\nCSVs -> {out}")


if __name__ == "__main__":
    main()
