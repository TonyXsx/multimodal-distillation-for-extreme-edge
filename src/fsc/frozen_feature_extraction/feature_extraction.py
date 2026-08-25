"""
Pulls pooled hidden states out of the frozen teacher for FSC.

Runs Qwen2.5-Omni-3B (4-bit NF4) over the small ablation subsets from
experiment_data_construction.py and saves a bank of pooled vectors, so the
probe and KD experiments later don't need the teacher forward pass again.

    selected_layers = [0, 9, 18, 24, 27, 30, 34, 36]
    llm_layers      = [9, 18, 24, 27, 30, 34, 36]

Two input orders, one forward pass each:

    prompt_first  [prompt] + [audio]
    audio_first   [audio] + [prompt]

44 pooled vectors per sample, fp16:

    layer 0       projected_audio_mean, projected_audio_last            2
    prompt_first  L{L}_audio_mean, L{L}_audio_last                     14
    audio_first   L{L}_audio_mean, L{L}_audio_last, L{L}_last_text,
                  L{L}_last_4_text_mean                                28

Layer 0 is the same whichever order you use, since the audio encoder never
attends to the text, so it's only taken from the prompt_first pass.

Each .pt is a dict with labels, sample_ids, metadata, feature_dim and a
features dict of [N, hidden_dim] fp16 tensors.
"""

import argparse
import gc
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# register the ffmpeg dlls before torch/soundfile touch any audio (windows)
if sys.platform == "win32":
    _ffmpeg_dll_dir = None
    for _p in os.environ.get("PATH", "").split(";"):
        if _p and os.path.exists(os.path.join(_p, "avcodec-62.dll")):
            _ffmpeg_dll_dir = os.add_dll_directory(_p)  # keep the ref, GC drops the dir
            break

import soundfile as sf
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT   # noqa: E402

SUBSET_DIR   = DATA_ROOT / "fsc_small_ablation"
OUT_DIR      = DATA_ROOT / "teacher_features" / "fsc_small_ablation__qwen2.5-omni-3b-4bit"


MODEL_NAME   = "Qwen/Qwen2.5-Omni-3B"
LLM_LAYERS   = [9, 18, 24, 27, 30, 34, 36]          # index into hidden_states
ADD_GEN_PROMPT = True                                # same as the teacher notebook

SHARD_SIZE       = 50    # flush every N samples, keeps the buffer small and lets it resume
EMPTY_CACHE_EVERY = 10   # empty_cache + gc every N, vram creeps up otherwise

# fallbacks, checked against the model config. overridden if the config has them
AUDIO_START_ID_DEFAULT = 151647
AUDIO_END_ID_DEFAULT   = 151648

TASK_PROMPT = (
    "You are classifying spoken smart-home commands. "
    "Each command has an action (e.g. activate/deactivate/increase), "
    "an object (e.g. lights/music/heat), and a location (e.g. bedroom/kitchen/none). "
    "Listen to the audio and describe the spoken command briefly."
)


def load_teacher():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


def get_special_id(model, name, default):
    """special-token id out of the config, which is sometimes nested. falls back
    to the hardcoded default."""
    cfg = model.config
    if hasattr(cfg, name):
        return getattr(cfg, name)
    for sub in ("thinker_config", "talker_config"):
        if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), name):
            return getattr(getattr(cfg, sub), name)
    return default


def find_audio_indices(input_ids, start_id, end_id):
    """positions of the audio placeholder tokens, between the start/end markers."""
    ids = input_ids[0].tolist()
    s = ids.index(start_id)
    e = ids.index(end_id)
    audio_idx = list(range(s + 1, e))
    if not audio_idx:
        raise ValueError("No audio tokens found between audio start/end markers.")
    return audio_idx, s, e


def _to_cpu_fp16(t):
    return t.detach().float().cpu().to(torch.float16)


def pool_prompt_first(hidden_states, audio_idx):
    """layer-0 projected audio, plus the prompt_first audio-token features."""
    feats = {}

    # layer 0 is the projected audio, before any LLM block, so order doesn't matter
    h0 = hidden_states[0][0]                                   # [seq, dim]
    feats["projected_audio_mean"] = _to_cpu_fp16(h0[audio_idx].mean(dim=0))
    feats["projected_audio_last"] = _to_cpu_fp16(h0[audio_idx[-1]])

    for L in LLM_LAYERS:
        h = hidden_states[L][0]
        feats[f"prompt_first_l{L}_audio_mean"] = _to_cpu_fp16(h[audio_idx].mean(dim=0))
        feats[f"prompt_first_l{L}_audio_last"] = _to_cpu_fp16(h[audio_idx[-1]])
    return feats


def pool_audio_first(hidden_states, audio_idx, seq_len):
    """the audio_first controls and the trailing-text features."""
    feats = {}
    last_text_idx = seq_len - 1
    last4_idx = list(range(max(0, seq_len - 4), seq_len))

    for L in LLM_LAYERS:
        h = hidden_states[L][0]
        feats[f"audio_first_l{L}_audio_mean"]       = _to_cpu_fp16(h[audio_idx].mean(dim=0))
        feats[f"audio_first_l{L}_audio_last"]       = _to_cpu_fp16(h[audio_idx[-1]])
        feats[f"audio_first_l{L}_last_text"]        = _to_cpu_fp16(h[last_text_idx])
        feats[f"audio_first_l{L}_last_4_text_mean"] = _to_cpu_fp16(h[last4_idx].mean(dim=0))
    return feats


def build_inputs(processor, model, audio_np, order):
    """order is 'prompt_first' or 'audio_first'."""
    audio_part = {"type": "audio", "audio": audio_np}
    text_part  = {"type": "text", "text": TASK_PROMPT}
    content = [text_part, audio_part] if order == "prompt_first" else [audio_part, text_part]

    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=ADD_GEN_PROMPT
    )
    inputs = processor(
        text=text_input, audio=[audio_np], sampling_rate=16000, return_tensors="pt"
    ).to(model.device)
    return inputs


@torch.no_grad()
def extract_sample(model, processor, audio_np, audio_start_id, audio_end_id):
    """(features, meta) for one sample."""
    feats, meta = {}, {}

    # prompt_first pass, gives layer-0 and the prompt_first audio features
    inputs_pf = build_inputs(processor, model, audio_np, "prompt_first")
    out_pf = model.thinker(**inputs_pf, output_hidden_states=True, return_dict=True)
    audio_idx_pf, s_pf, e_pf = find_audio_indices(inputs_pf["input_ids"], audio_start_id, audio_end_id)
    seq_pf = inputs_pf["input_ids"].shape[1]
    feats.update(pool_prompt_first(out_pf.hidden_states, audio_idx_pf))

    # audio_first pass, gives the audio controls and the trailing-text features
    inputs_af = build_inputs(processor, model, audio_np, "audio_first")
    out_af = model.thinker(**inputs_af, output_hidden_states=True, return_dict=True)
    audio_idx_af, s_af, e_af = find_audio_indices(inputs_af["input_ids"], audio_start_id, audio_end_id)
    seq_af = inputs_af["input_ids"].shape[1]
    feats.update(pool_audio_first(out_af.hidden_states, audio_idx_af, seq_af))

    meta["seq_len_prompt_first"]   = seq_pf
    meta["seq_len_audio_first"]    = seq_af
    meta["num_audio_tokens"]       = len(audio_idx_pf)
    meta["audio_range_prompt_first"] = [s_pf + 1, e_pf]
    meta["audio_range_audio_first"]  = [s_af + 1, e_af]
    meta["num_text_tokens_audio_first"] = seq_af - e_af - 1  # tokens after the audio
    return feats, meta


def _flush_shard(shard_dir, shard_idx, feat_bank, labels, sample_ids, metadata):
    """temp file then replace, so a kill mid-write can't leave a broken shard."""
    shard = {
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": sample_ids,
        "metadata": metadata,
        "features": {k: torch.stack(v, dim=0) for k, v in feat_bank.items()},
    }
    final_path = shard_dir / f"shard_{shard_idx:05d}.pt"
    tmp_path = shard_dir / f"shard_{shard_idx:05d}.pt.tmp"
    torch.save(shard, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def _load_shards(shard_dir):
    """load every shard in order and merge them back into one dict."""
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No shards found in {shard_dir}")

    feat_bank: dict[str, list] = {}
    labels, sample_ids, metadata = [], [], []
    for sp in shard_paths:
        sh = torch.load(sp, weights_only=False)
        labels.append(sh["labels"])
        sample_ids.extend(sh["sample_ids"])
        metadata.extend(sh["metadata"])
        for k, v in sh["features"].items():
            feat_bank.setdefault(k, []).append(v)

    features = {k: torch.cat(v, dim=0) for k, v in feat_bank.items()}
    return {
        "labels": torch.cat(labels, dim=0),
        "sample_ids": sample_ids,
        "metadata": metadata,
        "feature_dim": next(iter(features.values())).shape[1],
        "features": features,
    }


def _count_done(shard_dir):
    """how many samples are already on disk, for resuming."""
    done = 0
    for sp in sorted(shard_dir.glob("shard_*.pt")):
        done += len(torch.load(sp, weights_only=False)["sample_ids"])
    return done


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

    feat_bank: dict[str, list] = {}
    labels, sample_ids, metadata = [], [], []

    for i in tqdm(range(done, n), desc="extracting", initial=done, total=n):
        ex = ds[i]
        audio_np, _sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32", always_2d=False)

        feats, meta = extract_sample(model, processor, audio_np, audio_start_id, audio_end_id)

        for k, v in feats.items():
            feat_bank.setdefault(k, []).append(v)

        labels.append(int(ex["label_id"]))
        sample_ids.append(str(ex.get("file", i)))
        meta.update({
            "dataset_index": i,
            "intent": ex.get("intent"),
            "label_id": int(ex["label_id"]),
            "speaker_id": ex.get("speaker_id"),
        })
        metadata.append(meta)

        # release cached gpu memory now and then, it fragments on long runs
        if (i + 1) % EMPTY_CACHE_EVERY == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # flush once the buffer is full
        if len(labels) >= SHARD_SIZE:
            _flush_shard(shard_dir, next_shard_idx, feat_bank, labels, sample_ids, metadata)
            next_shard_idx += 1
            feat_bank, labels, sample_ids, metadata = {}, [], [], []

    # last partial buffer
    if labels:
        _flush_shard(shard_dir, next_shard_idx, feat_bank, labels, sample_ids, metadata)

    return _load_shards(shard_dir)


SPLITS = {
    "train": ("train_20pc", "train_20pc_features.pt"),
    "val":   ("val_10pc",   "val_10pc_features.pt"),
}


def main():
    parser = argparse.ArgumentParser(description="Extract frozen-teacher hidden features for FSC.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N samples (smoke test).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading teacher: {MODEL_NAME} (4-bit NF4) ...")
    model, processor = load_teacher()
    audio_start_id = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    audio_end_id   = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)
    print(f"audio_start_token_id={audio_start_id}  audio_end_token_id={audio_end_id}")

    targets = ["train", "val"] if args.split == "all" else [args.split]

    for split in targets:
        subset_name, out_name = SPLITS[split]
        ds = load_from_disk(str(SUBSET_DIR / subset_name))
        print(f"\n=== {split}: {subset_name}  ({len(ds)} samples) ===")

        shard_dir = OUT_DIR / f"{subset_name}_shards"
        result = process_split(model, processor, ds, audio_start_id, audio_end_id,
                               shard_dir, limit=args.limit)

        out_path = OUT_DIR / out_name
        torch.save(result, out_path)
        print(f"Saved {len(result['labels'])} samples x {len(result['features'])} features "
              f"(dim={result['feature_dim']}) -> {out_path}")

    # extraction config, so this can be reproduced and loaded later
    config = {
        "model_name": MODEL_NAME,
        "quantization": "4bit-nf4 (bnb, double-quant, fp16 compute)",
        "attn_implementation": "eager",
        "selected_layers": [0] + LLM_LAYERS,
        "llm_layers": LLM_LAYERS,
        "prompt_orders": ["prompt_first", "audio_first"],
        "add_generation_prompt": ADD_GEN_PROMPT,
        "task_prompt": TASK_PROMPT,
        "audio_start_token_id": int(audio_start_id),
        "audio_end_token_id": int(audio_end_id),
        "feature_dtype": "float16",
        "feature_naming": {
            "layer0": ["projected_audio_mean", "projected_audio_last"],
            "prompt_first": ["prompt_first_l{L}_audio_mean", "prompt_first_l{L}_audio_last"],
            "audio_first": [
                "audio_first_l{L}_audio_mean", "audio_first_l{L}_audio_last",
                "audio_first_l{L}_last_text", "audio_first_l{L}_last_4_text_mean",
            ],
        },
        "features_per_sample": 2 + len(LLM_LAYERS) * 2 + len(LLM_LAYERS) * 4,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT_DIR / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_DIR / 'extraction_config.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
