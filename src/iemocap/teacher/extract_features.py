"""
Extract teacher KD targets for IEMOCAP -- frozen and LoRA-adapted, one script.

Both arms run through the SAME code path, the same prompt and the same input
ordering (`iemocap.teacher.data`), and both load the backbone at the same
precision as the fine-tune. The only difference between them is
whether the LoRA weights are applied. That is what makes the frozen arm a
genuine control: on MIntRec the equivalent comparison (frozen audio probe
0.5443 -> QLoRA 0.5533, against 0.6130 for the readout token) is what showed
the adaptation's benefit lands in the transcript-conditioned readout rather
than in the audio representation, and the same question has to be answered
here before the LoRA step can be justified in the write-up.

Per utterance, in ONE forward pass:

    last_token           [H]  readout hidden state, final layer -- primary Feature-KD target
    audio_mean_final     [H]  clean audio-token mean, final layer
    audio_mean_l{L}      [H]  clean audio-token mean at layers 24/27/30/34
    audio_mean_L24-...   [H]  mean over those layers (the FSC-best combination)
    logits               [C]  classification head output -- Logit-KD target (adapted arm only)

`audio_mean*` is clean because audio precedes the transcript in the input, so
under causal masking those tokens never attend to the text (see
`iemocap.teacher.data`). `last_token` has seen everything -- that asymmetry is
the privileged-information story this track is testing.

Sharding, resume and the on-disk shard format are reused from the MIntRec
extractor, so the existing probe code can read these files unchanged.

Usage:
    # frozen control (no adapter)
    python src/iemocap/teacher/extract_features.py --split all
    # smoke first
    python src/iemocap/teacher/extract_features.py --split val --limit 5

    # after the fine-tune, the adapted arm
    python src/iemocap/teacher/extract_features.py --split all \
        --adapter data/iemocap/teacher_qlora/3b_bf16_audio-tr_r32a64/adapter_ep3 \
        --head    data/iemocap/teacher_qlora/3b_bf16_audio-tr_r32a64/head_ep3.pt
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from iemocap.paths import IEMOCAP_FEATURES  # noqa: E402
from iemocap.teacher.data import (  # noqa: E402
    CLASSES, LABEL2ID, INSTRUCTION, READOUT_CUE, SPLITS,
    load_split, wav_path, load_audio, build_inputs, input_order_str,
)
# Sharding / resume / layer choice / special-token lookup reused as-is.
from mintrec.teacher_probe.extract_features_local import (  # noqa: E402
    LLM_LAYERS, MEAN_COMBO, SHARD_SIZE, EMPTY_CACHE_EVERY,
    AUDIO_START_ID_DEFAULT, AUDIO_END_ID_DEFAULT,
    get_special_id, find_audio_indices, _cpu16,
    _flush_shard, _load_shards, _count_done,
)
# Same backbone loader the fine-tune uses, so the two cannot diverge.
from iemocap.teacher.backbone import MODELS, load_thinker, get_hidden_size  # noqa: E402
from mintrec.teacher_probe.qlora_finetune import OmniClassifier  # noqa: E402

SPLIT_FILE = {s: f"{s}_features.pt" for s in SPLITS}


def build_feat_tag(model_key, dtype, adapter_dir, use_transcript):
    mod = "audio-tr" if use_transcript else "audio"
    base = f"iemocap4__qwen2.5-omni-{model_key}-{dtype}"
    if adapter_dir is None:
        return f"{base}__frozen__{mod}"
    return f"{base}-LORA__{adapter_dir.name}__{mod}"


@torch.no_grad()
def extract_one(thinker, head, proc, device, row, use_transcript, a0, a1):
    wav = load_audio(wav_path(row))
    if wav.size == 0:
        raise ValueError("empty audio")
    inp = build_inputs(proc, device, wav, row["transcript"], use_transcript)
    out = thinker(**inp, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states

    last = int(inp["attention_mask"].sum(1) - 1)
    last_feat = hs[-1][0, last]
    audio_idx, _, _ = find_audio_indices(inp["input_ids"], a0, a1)

    feats = {"last_token": _cpu16(last_feat),
             "audio_mean_final": _cpu16(hs[-1][0, audio_idx].mean(0))}
    per_layer = []
    for L in LLM_LAYERS:
        v = hs[L][0, audio_idx].mean(0)
        feats[f"audio_mean_l{L}"] = _cpu16(v)
        per_layer.append(v)
    feats[f"audio_mean_{MEAN_COMBO}"] = _cpu16(torch.stack(per_layer, 0).mean(0))
    if head is not None:
        feats["logits"] = _cpu16(head(last_feat.to(head[0].weight.dtype)))

    return feats, {"seq_len": int(inp["input_ids"].shape[1]),
                   "num_audio_tokens": len(audio_idx)}


def process_split(thinker, head, proc, device, df, use_transcript, a0, a1, shard_dir, limit):
    n = len(df) if limit is None else min(limit, len(df))
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = _count_done(shard_dir)
    if done >= n:
        print(f"  resume: {done} already extracted (>= {n}) -- merging.")
        return _load_shards(shard_dir)
    if done > 0:
        print(f"  resume: {done} already extracted -- continuing.")

    nxt = len(list(shard_dir.glob("shard_*.pt")))
    fb, labels, ids, meta = {}, [], [], []
    for i in tqdm(range(done, n), desc="extract", initial=done, total=n):
        row = df.iloc[i]
        try:
            feats, m = extract_one(thinker, head, proc, device, row, use_transcript, a0, a1)
        except Exception as ex:
            print(f"  skip {row['turn_id']}: {type(ex).__name__}: {ex}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            continue
        for k, v in feats.items():
            fb.setdefault(k, []).append(v)
        labels.append(int(row["label"]))
        ids.append(row["turn_id"])
        m.update({"id": row["turn_id"], "label": row["emotion"], "label_id": int(row["label"]),
                  "speaker": row["speaker"], "is_impro": bool(row["is_impro"]),
                  "duration": float(row["duration"])})
        meta.append(m)

        if (i + 1) % EMPTY_CACHE_EVERY == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        if len(labels) >= SHARD_SIZE:
            _flush_shard(shard_dir, nxt, fb, labels, ids, meta)
            nxt += 1
            fb, labels, ids, meta = {}, [], [], []

    if labels:
        _flush_shard(shard_dir, nxt, fb, labels, ids, meta)
    return _load_shards(shard_dir)


def main():
    ap = argparse.ArgumentParser(
        description="Extract IEMOCAP teacher features (frozen control or LoRA-adapted).")
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N samples (smoke test)")
    ap.add_argument("--model", choices=["3b", "7b"], default="3b")
    ap.add_argument("--dtype", choices=["bf16", "4bit"], default="bf16")
    ap.add_argument("--adapter", default=None,
                    help="path to adapter_epN/ -- omit for the FROZEN control arm")
    ap.add_argument("--head", default=None,
                    help="path to head_epN.pt (required with --adapter, gives the logits target)")
    ap.add_argument("--use-transcript", dest="use_transcript", action="store_true", default=True)
    ap.add_argument("--no-transcript", dest="use_transcript", action="store_false")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter) if args.adapter else None
    if adapter_dir is not None and args.head is None:
        raise SystemExit("--head is required with --adapter (it produces the logit-KD target)")

    model_name = MODELS[args.model]
    use_transcript, dtype = args.use_transcript, args.dtype
    # An adapted run must reproduce the fine-tune exactly -- including its
    # precision, since an adapter trained on a bf16 base is not valid on a
    # 4-bit one (and vice versa).
    if adapter_dir is not None:
        cfg_path = adapter_dir.parent / "config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            model_name = cfg.get("model_name", model_name)
            use_transcript = bool(cfg.get("use_transcript", use_transcript))
            dtype = cfg.get("dtype", dtype)
            print(f"reproducing training config from {cfg_path.name}: "
                  f"dtype={dtype}, use_transcript={use_transcript}")
        else:
            print(f"WARNING: no config.json beside {adapter_dir} -- using CLI flags as given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} ({dtype}) ...", flush=True)
    model, proc = load_thinker(model_name, dtype)
    hidden = get_hidden_size(model)
    a0 = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    a1 = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)

    thinker, head = model.thinker, None
    if adapter_dir is not None:
        from peft import PeftModel
        thinker = PeftModel.from_pretrained(model.thinker, str(adapter_dir))
        head = OmniClassifier(thinker, hidden, len(CLASSES)).head.to(device)
        head.load_state_dict(torch.load(args.head, map_location=device))
        head.to(torch.bfloat16).eval()
    thinker.eval()

    tag = build_feat_tag(args.model, dtype, adapter_dir, use_transcript)
    out_dir = IEMOCAP_FEATURES / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ARM      : {'LoRA-adapted' if adapter_dir else 'FROZEN control'}")
    print(f"FEAT_TAG : {tag}")
    print(f"OUT      : {out_dir}")
    print(f"ORDER    : {input_order_str(use_transcript)}")

    targets = list(SPLITS) if args.split == "all" else [args.split]
    for split in targets:
        df = load_split(split)
        print(f"\n=== {split}: {len(df)} utterances ===")
        res = process_split(thinker, head, proc, device, df, use_transcript,
                            a0, a1, out_dir / f"{split}_shards", args.limit)
        torch.save(res, out_dir / SPLIT_FILE[split])
        print(f"saved {len(res['labels'])} x {len(res['features'])} tensors "
              f"(keys={sorted(res['features'])}) -> {out_dir / SPLIT_FILE[split]}")

    config = {
        "dataset": "IEMOCAP 4-class (exc merged into hap)",
        "arm": "qlora" if adapter_dir else "frozen",
        "source_adapter": str(adapter_dir) if adapter_dir else None,
        "source_head": args.head,
        "model_name": model_name, "precision": dtype, "hidden": hidden,
        "use_transcript": use_transcript, "input_order": input_order_str(use_transcript),
        "instruction": INSTRUCTION, "readout_cue": READOUT_CUE,
        "llm_layers": LLM_LAYERS, "mean_combo": f"audio_mean_{MEAN_COMBO}",
        "feature_keys": sorted(res["features"]),
        "n_classes": len(CLASSES), "classes": CLASSES, "label2id": LABEL2ID,
        "feature_dtype": "float16",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_dir / 'extraction_config.json'}\nDone.")


if __name__ == "__main__":
    main()
