"""
Extract KD targets from the QLoRA-adapted teacher (LOCAL-friendly).

After ONE QLoRA fine-tune (qlora_finetune.py, run on RunPod), copy the small
adapter folder back to this machine and run this script. It loads the local base
Qwen2.5-Omni-3B (already in hf_cache) + the LoRA adapter + the trained head, then
for every MIntRec2.0 utterance dumps, in ONE forward pass:

    logits            [30]    -> logit-KD target  (head on the readout token)
    last_token        [H]     -> the readout representation (final layer)
    audio_mean_final  [H]     -> CLEAN audio feature (audio block, final layer)
    audio_mean_l27    [H]     -> clean audio feature at layer 27 (frozen-best layer)

Input order is reconstructed IDENTICALLY to training from the saved config.json
(instruction -> audio -> video -> transcript -> "Intent:"). Fits the 6 GB laptop
(4-bit, sub-sampled frames). Sharded + resume-safe.

Usage:
    python src/mintrec/teacher_probe/extract_with_lora.py \
        --adapter data/mintrec/teacher_qlora/3b_tva_tr_last_r32/adapter_ep3 \
        --head    data/mintrec/teacher_qlora/3b_tva_tr_last_r32/head_ep3.pt \
        --split all
    # smoke first:  --split dev --limit 5
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm
from peft import PeftModel

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA  # noqa: E402
from mintrec.teacher_probe.extract_features_local import (  # noqa: E402
    load_split_df, build_label2id, build_stem2path, find_video,
    extract_frames, load_audio, get_special_id, ANNO_DIR,
    _flush_shard, _load_shards, _count_done,
)
from mintrec.teacher_probe.qlora_finetune import (  # noqa: E402
    load_backbone, get_hidden_size, build_inputs, OmniClassifier, MODELS,
    AUDIO_START_ID_DEFAULT, AUDIO_END_ID_DEFAULT,
)

L27 = 27
SHARD_SIZE = 50


def _cpu16(t):
    return t.detach().float().cpu().to(torch.float16)


@torch.no_grad()
def extract_one(thinker, head, proc, device, wav, frames, transcript, args, a0, a1):
    inp = build_inputs(proc, device, wav, frames, transcript, args)
    out = thinker(**inp, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    am = inp["attention_mask"]
    last = int(am.sum(1) - 1)
    ids = inp["input_ids"][0].tolist()
    s, e = ids.index(a0), ids.index(a1)
    audio_idx = list(range(s + 1, e))
    last_feat = hs[-1][0, last]
    feats = {
        "logits": _cpu16(head(last_feat.to(head[0].weight.dtype))),
        "last_token": _cpu16(last_feat),
        "audio_mean_final": _cpu16(hs[-1][0, audio_idx].mean(0)),
        "audio_mean_l27": _cpu16(hs[L27][0, audio_idx].mean(0)),
    }
    return feats, {"seq_len": len(ids), "num_audio_tokens": len(audio_idx)}


def process_split(thinker, head, proc, device, df, s2p, label2id, args, a0, a1, shard_dir):
    n = len(df) if args.limit is None else min(args.limit, len(df))
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = _count_done(shard_dir)
    if done >= n:
        print(f"  resume: {done} done (>= {n}) — merging."); return _load_shards(shard_dir)
    if done > 0:
        print(f"  resume from {done}")
    nxt = len(list(shard_dir.glob("shard_*.pt")))
    fb, labels, ids, meta = {}, [], [], []
    for i in tqdm(range(done, n), desc="extract", initial=done, total=n):
        row = df.iloc[i]
        vp = find_video(row, s2p)
        if vp is None:
            continue
        try:
            wav = load_audio(vp)
            if wav.size == 0:
                continue
            frames = extract_frames(vp, args.frames) if args.modalities == "tva" else None
            feats, m = extract_one(thinker, head, proc, device, wav, frames, row["text"], args, a0, a1)
        except Exception as ex:
            print("  skip", row["id"], type(ex).__name__, ex)
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); gc.collect()
            continue
        for k, v in feats.items():
            fb.setdefault(k, []).append(v)
        labels.append(label2id[row["label"]]); ids.append(row["id"])
        m.update({"id": row["id"], "label": row["label"], "label_id": label2id[row["label"]]}); meta.append(m)
        if (i + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()
        if len(labels) >= SHARD_SIZE:
            _flush_shard(shard_dir, nxt, fb, labels, ids, meta); nxt += 1
            fb, labels, ids, meta = {}, [], [], []
    if labels:
        _flush_shard(shard_dir, nxt, fb, labels, ids, meta)
    return _load_shards(shard_dir)


SPLIT_FILE = {"train": "train_features.pt", "dev": "dev_features.pt", "test": "test_features.pt"}


def main():
    ap = argparse.ArgumentParser(description="Extract KD targets from the QLoRA-adapted teacher.")
    ap.add_argument("--adapter", required=True, help="path to adapter_epN/ (LoRA weights)")
    ap.add_argument("--head", required=True, help="path to head_epN.pt")
    ap.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    args_cli = ap.parse_args()

    adapter_dir = Path(args_cli.adapter)
    cfg = json.load(open(adapter_dir.parent / "config.json", encoding="utf-8"))
    # reconstruct the exact training input config
    args = SimpleNamespace(frames=cfg["frames"], modalities=cfg["modalities"],
                           use_transcript=cfg["use_transcript"], limit=args_cli.limit)
    model_name = cfg["model_name"]
    label2id = {k: int(v) for k, v in cfg["label2id"].items()}
    n_classes = cfg["n_classes"]
    print(f"base={model_name} | adapter={adapter_dir.name} | order={cfg.get('input_order')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_backbone(model_name)
    hidden = get_hidden_size(model)
    a0 = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    a1 = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)
    thinker = PeftModel.from_pretrained(model.thinker, str(adapter_dir)).eval()

    head = OmniClassifier(thinker, hidden, n_classes).head.to(device)
    head.load_state_dict(torch.load(args_cli.head, map_location=device))
    head.to(torch.bfloat16).eval()

    tag = (f"mintrec2.0__{Path(model_name).name.lower()}-4bit-QLORA__"
           f"{cfg['modalities']}_{'tr' if cfg['use_transcript'] else 'notr'}__{adapter_dir.name}")
    out_dir = MINTREC_DATA / "teacher_features" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_dir}")

    s2p, n_mp4 = build_stem2path()
    dfs = {s: load_split_df(s) for s in ("train", "dev", "test") if (ANNO_DIR / f"{s}.tsv").exists()}
    targets = list(dfs) if args_cli.split == "all" else [args_cli.split]
    for split in targets:
        print(f"\n=== {split}: {len(dfs[split])} ===")
        res = process_split(thinker, head, proc, device, dfs[split], s2p, label2id, args, a0, a1,
                            out_dir / f"{split}_shards")
        torch.save(res, out_dir / SPLIT_FILE[split])
        print(f"saved {len(res['labels'])} x {len(res['features'])} tensors "
              f"(keys={list(res['features'])}) -> {out_dir / SPLIT_FILE[split]}")

    with open(out_dir / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump({"source_adapter": str(adapter_dir), "base": model_name, "from_qlora_config": cfg,
                   "feature_keys": ["logits", "last_token", "audio_mean_final", "audio_mean_l27"],
                   "n_classes": n_classes, "label2id": label2id}, f, indent=2, ensure_ascii=False)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
