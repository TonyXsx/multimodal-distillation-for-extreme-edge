"""
FULL-dataset frozen-teacher feature extraction — single KD-target feature.

Based on the two ablation rounds, the chosen teacher signal is:

    prompt_first · audio_mean · mean over layers [24, 27, 30, 34]
    (val acc 0.923 on the small ablation — the strongest audio-derived feature)

This script runs frozen Qwen2.5-Omni-3B (4-bit NF4) over the FULL FSC dataset
and stores, per sample, exactly ONE pooled [2048] float16 vector — not all the
hidden states. Reuses model-loading / token-finding / sharded-resume logic from
`feature_extraction.py`.

Pooling (identical to the 'mean[24,27,30,34]' combo that won the ablation):
    for each L in [24, 27, 30, 34]:  v_L = mean over audio-token hidden states at layer L
    feature = mean(v_24, v_27, v_30, v_34)              -> [2048]

Splits:
    train : FSC train      (23,132 samples)  -> KD training target
    val   : FSC validation ( 3,118 samples)  -> monitors student KD losses
    test  : NOT extracted. The student is audio-only and must be evaluated on
            raw test audio without any teacher signal — extracting teacher
            features on test would leak information.

Output (clear naming, same convention as the ablation bank):
    data/teacher_features/fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34/
        train_features.pt
        val_features.pt
        train_shards/ , val_shards/      (checkpoints; deletable after merge)
        extraction_config.json

Each .pt is a dict (same format as the ablation files, single feature key):
    {
        "labels": LongTensor[N],
        "sample_ids": list[str],
        "metadata": list[dict],
        "feature_dim": 2048,
        "features": {"prompt_first_audio_mean_L24-27-30-34": FloatTensor[N,2048] fp16},
    }

Runtime: ~14-15h on a laptop 3060 (1 forward/sample). Resume-safe — leave it
running overnight; an interruption resumes from the last completed 50-sample shard.
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the sibling module importable regardless of CWD, then reuse its helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import (          # noqa: E402  (import after sys.path tweak)
    MODEL_NAME,
    _count_done,
    _flush_shard,
    _load_shards,
    build_inputs,
    find_audio_indices,
    get_special_id,
    load_teacher,
)

import torch                              # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from tqdm import tqdm                     # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT   # noqa: E402

LABEL_CONFIG = DATA_ROOT / "fsc_small_ablation" / "config.json"   # reuse identical label2id
OUT_DIR      = DATA_ROOT / "teacher_features" / "fsc_full__qwen2.5-omni-3b-4bit__pf_audiomean_L24-27-30-34"

# ── Feature config ──────────────────────────────────────────────────────────────
COMBINE_LAYERS = [24, 27, 30, 34]
FEATURE_NAME   = "prompt_first_audio_mean_L24-27-30-34"

SHARD_SIZE        = 50
EMPTY_CACHE_EVERY = 10

AUDIO_START_ID_DEFAULT = 151647
AUDIO_END_ID_DEFAULT   = 151648


# ── Single-sample extraction (one prompt_first forward -> one pooled vector) ─────────
@torch.no_grad()
def extract_sample(model, processor, audio_np, audio_start_id, audio_end_id):
    inputs = build_inputs(processor, model, audio_np, "prompt_first")
    out = model.thinker(**inputs, output_hidden_states=True, return_dict=True)

    audio_idx, s, e = find_audio_indices(inputs["input_ids"], audio_start_id, audio_end_id)
    seq_len = inputs["input_ids"].shape[1]
    hs = out.hidden_states

    # Per-layer audio-token mean (in fp32 for stable averaging), then mean across layers.
    layer_means = [hs[L][0][audio_idx].float().mean(dim=0) for L in COMBINE_LAYERS]
    combined = torch.stack(layer_means, dim=0).mean(dim=0)        # [hidden_dim]
    feat = combined.cpu().to(torch.float16)

    meta = {
        "seq_len": seq_len,
        "num_audio_tokens": len(audio_idx),
        "audio_range_prompt_first": [s + 1, e],
    }
    return feat, meta


# ── Split processing (sharded + resume; mirrors feature_extraction.process_split) ────
def process_split(model, processor, ds, audio_start_id, audio_end_id, shard_dir, limit=None):
    n = len(ds) if limit is None else min(limit, len(ds))
    shard_dir.mkdir(parents=True, exist_ok=True)

    done = _count_done(shard_dir)
    if done >= n:
        print(f"  resume: {done} samples already extracted (>= {n}) — skipping to merge.")
        return _load_shards(shard_dir)
    if done > 0:
        print(f"  resume: {done} samples already extracted — continuing from index {done}.")

    next_shard_idx = len(list(shard_dir.glob("shard_*.pt")))

    feat_bank, labels, sample_ids, metadata = {}, [], [], []

    for i in tqdm(range(done, n), desc="extracting", initial=done, total=n):
        ex = ds[i]
        audio_np, _sr = sf_read(ex["audio"]["bytes"])

        feat, meta = extract_sample(model, processor, audio_np, audio_start_id, audio_end_id)
        feat_bank.setdefault(FEATURE_NAME, []).append(feat)

        labels.append(int(ex["label_id"]))
        sample_ids.append(str(ex.get("file", i)))
        meta.update({
            "dataset_index": i,
            "intent": ex.get("intent"),
            "label_id": int(ex["label_id"]),
            "speaker_id": ex.get("speaker_id"),
        })
        metadata.append(meta)

        if (i + 1) % EMPTY_CACHE_EVERY == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        if len(labels) >= SHARD_SIZE:
            _flush_shard(shard_dir, next_shard_idx, feat_bank, labels, sample_ids, metadata)
            next_shard_idx += 1
            feat_bank, labels, sample_ids, metadata = {}, [], [], []

    if labels:
        _flush_shard(shard_dir, next_shard_idx, feat_bank, labels, sample_ids, metadata)

    return _load_shards(shard_dir)


# soundfile decode helper (kept local; feature_extraction imports sf at module level too)
import soundfile as sf  # noqa: E402

def sf_read(raw_bytes):
    return sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)


# ── FSC loading (full dataset + intent/label_id, reusing the ablation's label2id) ────
def load_full_fsc():
    with open(LABEL_CONFIG, encoding="utf-8") as f:
        label2id = json.load(f)["label2id"]

    fsc = load_dataset("s3prl/superb", name="ic", cache_dir=str(DATA_ROOT))
    feats = fsc["train"].features
    action_names, object_names, location_names = (
        feats["action"].names, feats["object"].names, feats["location"].names
    )
    fsc = fsc.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def add_meta(ex):
        intent = (f"{action_names[ex['action']]}_"
                  f"{object_names[ex['object']]}_"
                  f"{location_names[ex['location']]}")
        ex["intent"] = intent
        ex["label_id"] = label2id[intent]
        return ex

    fsc = fsc.map(add_meta, desc="Adding intent/label_id")
    return fsc, label2id


# ── Main ──────────────────────────────────────────────────────────────────────────
SPLITS = {
    "train": ("train",      "train_features.pt", "train_shards"),
    "val":   ("validation", "val_features.pt",   "val_shards"),
}


def main():
    parser = argparse.ArgumentParser(description="Full-FSC single-feature teacher extraction.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N (smoke test).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading teacher: {MODEL_NAME} (4-bit NF4) ...")
    model, processor = load_teacher()
    audio_start_id = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    audio_end_id   = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)
    print(f"audio_start_token_id={audio_start_id}  audio_end_token_id={audio_end_id}")

    print("Loading full FSC ...")
    fsc, label2id = load_full_fsc()

    targets = ["train", "val"] if args.split == "all" else [args.split]

    for split in targets:
        hf_split, out_name, shard_name = SPLITS[split]
        ds = fsc[hf_split]
        print(f"\n=== {split}: FSC.{hf_split}  ({len(ds)} samples) ===")

        shard_dir = OUT_DIR / shard_name
        result = process_split(model, processor, ds, audio_start_id, audio_end_id,
                               shard_dir, limit=args.limit)

        out_path = OUT_DIR / out_name
        torch.save(result, out_path)
        print(f"Saved {len(result['labels'])} samples x {len(result['features'])} feature "
              f"(dim={result['feature_dim']}) -> {out_path}")

    config = {
        "model_name": MODEL_NAME,
        "quantization": "4bit-nf4 (bnb, double-quant, fp16 compute)",
        "attn_implementation": "eager",
        "source_dataset": "s3prl/superb (ic) — FULL train + validation",
        "prompt_order": "prompt_first",
        "combine_layers": COMBINE_LAYERS,
        "pooling": "audio-token mean per layer, then mean across layers",
        "feature_name": FEATURE_NAME,
        "feature_dim": 2048,
        "feature_dtype": "float16",
        "n_classes": len(label2id),
        "label2id": label2id,
        "audio_start_token_id": int(audio_start_id),
        "audio_end_token_id": int(audio_end_id),
        "test_extracted": False,
        "test_note": "Student is audio-only; test must be evaluated on raw audio without teacher.",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_DIR / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_DIR / 'extraction_config.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
