"""
LoRA fine-tune Qwen2.5-Omni (Thinker) for IEMOCAP 4-class emotion recognition.

bf16 base by default -- see `backbone.py` for why this replaces MIntRec's
4-bit setup and why the LoRA coverage differs (audio tower fully adapted and
at a higher rank, visual tower excluded).

Reused unchanged from the MIntRec track: `OmniClassifier` (the pooled-readout
classification wrapper). Everything else here is IEMOCAP-specific.

Selection is by validation UNWEIGHTED accuracy (macro recall), not plain
accuracy: the splits have visibly different class priors (validation is 35.6%
happy against 13.7% angry), so plain accuracy partly rewards following the
prior.

Memory at bf16 on a 24 GB card: ~9.4 GB frozen weights, ~1.5 GB adapter grads
and optimiser state, 1-2 GB activations at batch 1 with gradient
checkpointing. Roughly 13 GB, leaving headroom.

Usage:
    python src/iemocap/teacher/lora_finetune.py --limit 40 --eval-limit 40 --epochs 1
    python src/iemocap/teacher/lora_finetune.py
    python src/iemocap/teacher/lora_finetune.py --no-transcript      # audio-only teacher
    python src/iemocap/teacher/lora_finetune.py --dtype 4bit         # small-GPU fallback
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, prepare_model_for_kbit_training

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_QLORA, IEMOCAP_OUTPUTS  # noqa: E402
from iemocap.teacher.backbone import (  # noqa: E402
    MODELS, LORA_TARGETS, LORA_EXCLUDE,
    load_thinker, get_hidden_size, build_lora_config, describe_trainable,
)
from iemocap.teacher.data import (  # noqa: E402
    CLASSES, LABEL2ID, INSTRUCTION, READOUT_CUE,
    load_split, wav_path, load_audio, build_inputs, input_order_str,
)
from mintrec.teacher_probe.qlora_finetune import OmniClassifier  # noqa: E402


def prep_sample(proc, device, row, use_transcript):
    wav = load_audio(wav_path(row))
    if wav.size == 0:
        return None
    return build_inputs(proc, device, wav, row["transcript"], use_transcript)


@torch.no_grad()
def evaluate(clf, proc, device, df, use_transcript, desc="eval"):
    clf.eval()
    ys, ps = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        try:
            inp = prep_sample(proc, device, row, use_transcript)
            if inp is None:
                continue
            logits = clf(inp)
        except Exception as ex:
            print("  eval skip", row["turn_id"], type(ex).__name__, ex)
            continue
        ps.append(int(logits.argmax(-1)[0]))
        ys.append(int(row["label"]))
    ys, ps = np.array(ys), np.array(ps)
    return {
        "n": int(len(ys)),
        "wa": float((ps == ys).mean()),                       # weighted acc == plain accuracy
        "ua": float(recall_score(ys, ps, average="macro")),   # unweighted acc == macro recall
        "macro_f1": float(f1_score(ys, ps, average="macro")),
        "per_class_recall": recall_score(ys, ps, average=None,
                                         labels=list(range(len(CLASSES)))).round(4).tolist(),
        "confusion": confusion_matrix(ys, ps, labels=list(range(len(CLASSES)))).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-Omni on IEMOCAP (4-class emotion).")
    ap.add_argument("--model", choices=["3b", "7b"], default="3b")
    ap.add_argument("--dtype", choices=["bf16", "4bit"], default="bf16")
    ap.add_argument("--use-transcript", dest="use_transcript", action="store_true", default=True)
    ap.add_argument("--no-transcript", dest="use_transcript", action="store_false",
                    help="audio-only teacher (modality ablation)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=16, help="grad-accumulation steps (physical batch=1)")
    ap.add_argument("--lora-r", type=int, default=32, help="rank for the LLM backbone")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--audio-r", type=int, default=64,
                    help="rank for the audio tower (0 = same as --lora-r)")
    ap.add_argument("--audio-alpha", type=int, default=128)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="first N train samples (smoke)")
    ap.add_argument("--eval-limit", type=int, default=None, help="first N val samples (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"
    model_name = MODELS[args.model]
    train_df = load_split("train", limit=args.limit)
    val_df = load_split("val", limit=args.eval_limit)
    n_classes = len(CLASSES)
    print(f"{n_classes} classes {CLASSES} | train {len(train_df)} | val {len(val_df)}")
    print(f"input order: {input_order_str(args.use_transcript)}")

    print(f"Loading {model_name} ({args.dtype}) ...", flush=True)
    model, proc = load_thinker(model_name, args.dtype)
    hidden = get_hidden_size(model)

    if args.dtype == "4bit":
        thinker = prepare_model_for_kbit_training(model.thinker, use_gradient_checkpointing=True)
    else:
        thinker = model.thinker
        thinker.gradient_checkpointing_enable()
        # Without this the checkpointed blocks see inputs that do not require
        # grad, so nothing flows back into the adapters.
        if hasattr(thinker, "enable_input_require_grads"):
            thinker.enable_input_require_grads()

    thinker = get_peft_model(thinker, build_lora_config(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        audio_r=args.audio_r, audio_alpha=args.audio_alpha))
    groups, n_train_p, n_total_p = describe_trainable(thinker)
    if hasattr(thinker, "config"):
        thinker.config.use_cache = False

    clf = OmniClassifier(thinker, hidden, n_classes, pool="last",
                         head_hidden=args.head_hidden).to(device)
    clf.head.to(torch.bfloat16)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([
        {"params": [p for p in thinker.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": clf.head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    steps = max((len(train_df) * args.epochs) // args.accum, 1)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

    tag = (f"{args.model}_{args.dtype}_audio{'-tr' if args.use_transcript else ''}"
           f"_r{args.lora_r}a{args.audio_r or args.lora_r}")
    out_dir = IEMOCAP_QLORA / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = IEMOCAP_OUTPUTS / "teacher_lora"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"lora_{tag}.csv"
    print(f"OUT: {out_dir}\nLOG: {log_csv}", flush=True)

    rows = []
    for ep in range(args.epochs):
        clf.train()
        opt.zero_grad()
        order = np.random.permutation(len(train_df))
        running, seen = 0.0, 0
        pbar = tqdm(order, desc=f"epoch {ep+1}/{args.epochs}")
        for step, ridx in enumerate(pbar):
            row = train_df.iloc[int(ridx)]
            try:
                inp = prep_sample(proc, device, row, args.use_transcript)
                if inp is None:
                    continue
                logits = clf(inp)
                y = torch.tensor([int(row["label"])], device=device)
                loss = loss_fn(logits, y) / args.accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                print("  OOM skip", row["turn_id"], f"dur={row['duration']:.1f}s")
                opt.zero_grad()
                torch.cuda.empty_cache()
                continue
            except Exception as ex:
                print("  skip", row["turn_id"], type(ex).__name__, ex)
                continue
            running += loss.item() * args.accum
            seen += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in thinker.parameters() if p.requires_grad]
                    + list(clf.head.parameters()), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                pbar.set_postfix(loss=f"{running/max(seen,1):.3f}",
                                 gb=f"{torch.cuda.max_memory_allocated()/1e9:.1f}")

        # True LOSO has no validation set. Train the fixed number of epochs and
        # keep them all; the caller decides which adapter to extract with.
        m = (evaluate(clf, proc, device, val_df, args.use_transcript)
             if len(val_df) else None)
        if m is None:
            print(f"[epoch {ep+1}] train_loss={running/max(seen,1):.4f}  "
                  f"(no val split in this protocol)  "
                  f"peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
        else:
            print(f"[epoch {ep+1}] train_loss={running/max(seen,1):.4f}  "
                  f"val WA={m['wa']:.4f}  UA={m['ua']:.4f}  macroF1={m['macro_f1']:.4f}  "
                  f"peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
            print(f"  per-class recall {dict(zip(CLASSES, m['per_class_recall']))}",
                  flush=True)
        thinker.save_pretrained(out_dir / f"adapter_ep{ep+1}")
        torch.save(clf.head.state_dict(), out_dir / f"head_ep{ep+1}.pt")
        row = {"epoch": ep + 1, "train_loss": round(running / max(seen, 1), 4),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        if m is not None:
            row.update({"val_n": m["n"], "val_wa": m["wa"], "val_ua": m["ua"],
                        "val_macro_f1": m["macro_f1"],
                        **{f"recall_{c}": v for c, v in zip(CLASSES, m["per_class_recall"])}})
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_csv, index=False)

    if any("val_ua" in r for r in rows):
        best = max(rows, key=lambda r: r.get("val_ua", -1))
        selection = "val_ua"
        print(f"\nBest epoch by val UA: {best['epoch']} (UA={best['val_ua']:.4f}) "
              f"-> extract with adapter_ep{best['epoch']} / head_ep{best['epoch']}.pt")
    else:
        best = rows[-1]
        selection = "none (no val split; last epoch)"
        print(f"\nNo val split -- nothing selected. Last epoch is {best['epoch']}; "
              f"extract with the adapter the caller asks for.")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "model_name": model_name, "hidden": hidden,
                   "n_classes": n_classes, "classes": CLASSES, "label2id": LABEL2ID,
                   "lora_targets": LORA_TARGETS, "lora_exclude": LORA_EXCLUDE,
                   "trainable_params": n_train_p, "total_params": n_total_p,
                   "instruction": INSTRUCTION, "readout_cue": READOUT_CUE,
                   "input_order": input_order_str(args.use_transcript),
                   "selection_metric": selection, "best_epoch": best["epoch"],
                   "epochs_log": rows}, f, indent=2, ensure_ascii=False)
    print(f"Saved adapters + heads + config -> {out_dir}")


if __name__ == "__main__":
    main()
