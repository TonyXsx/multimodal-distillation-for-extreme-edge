"""
Frozen multimodal-teacher hidden-representation extraction for MIntRec 2.0.

Quick validation of the PRIVILEGED / cross-modal distillation setup:
the teacher (Qwen2.5-Omni-3B, 4-bit) sees ALL THREE modalities, but we only
pool the AUDIO-token hidden states - so a downstream audio-only student has a
target it can (partially) reproduce while still benefiting from the video+text
context the teacher absorbed via attention.

Input layout (single forward pass, prompt_first):

    [ text (task prompt + transcript) ]  +  [ video frames ]  +  [ audio tokens ]
                                                                  ^^^^^^^^^^^^^^^^
    audio is placed LAST on purpose: under causal attention its token
    representations attend over the preceding text + video, so the pooled
    audio_mean vector carries the privileged multimodal information.

We feed video frames WITHOUT their audio track (use_audio_in_video=False) and
supply the clip's audio as a separate `audio` part, so the audio tokens form one
clean contiguous block at the end (locatable via the audio start/end markers).

Pooling (mirrors the FSC winner - prompt_first · audio_mean · mid/late layers):

    pf_audio_mean_l{L}            for L in [24, 27, 30, 34]
    pf_audio_mean_L24-27-30-34    mean over those four layers

Output (same format/habit as the FSC extractor):

    data/teacher_features/<FEAT_TAG>/
        train_features.pt
        dev_features.pt
        extraction_config.json

Each .pt is a dict:
    {
        "labels":      LongTensor [N],
        "sample_ids":  list[str],          # the MMLA 'id' = '{dia}_{utt}'
        "metadata":    list[dict],
        "feature_dim": int,
        "features":    {name: FloatTensor[N, hidden_dim] (float16), ...},
    }

Usage:
    python extract_features.py --split all              # train + dev, full
    python extract_features.py --split dev --limit 50   # smoke test
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path


if sys.platform == "win32":
    _ffmpeg_dll_dir = None
    for _p in os.environ.get("PATH", "").split(";"):
        if _p and os.path.exists(os.path.join(_p, "avcodec-62.dll")):
            _ffmpeg_dll_dir = os.add_dll_directory(_p)  # keep ref alive
            break

import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info


_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA   # noqa: E402


ANNO_DIR  = MINTREC_DATA / "MIntRec2.0"          # train/dev/test.tsv live here
VIDEO_DIR = ANNO_DIR / "video"                   # extracted .mp4 clips


def build_feat_tag(dtype):
    """FEAT_TAG carries the teacher precision so fp16/bf16/4bit runs never clash."""
    return f"mintrec2.0_multimodal__qwen2.5-omni-3b-{dtype}__pf_text-video-audio__audiomean"


MODEL_NAME    = "Qwen/Qwen2.5-Omni-3B"
LLM_LAYERS    = [24, 27, 30, 34]                 # FSC-winning mid/late layers
MEAN_COMBO    = "L24-27-30-34"
ADD_GEN_PROMPT = True
SHARD_SIZE        = 50
EMPTY_CACHE_EVERY = 5

AUDIO_START_ID_DEFAULT = 151647
AUDIO_END_ID_DEFAULT   = 151648

# Labels are derived from the TSVs at runtime (see build_label2id) rather than
# hard-coded - the MMLA label strings may differ in casing/wording from the
# official benchmark config, and we must not silently drop a whole class.

# Task prompt carries the transcript (the 'text' modality) + a brief framing.
TASK_PROMPT_TEMPLATE = (
    "You are analyzing a short TV-show clip to recognize the speaker's intent "
    "among 30 fine-grained intent classes. "
    'The transcript of what is said is: "{text}".'
)


def load_split_df(name):
    df = pd.read_csv(ANNO_DIR / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]            # id, text, label, dimension
    df["id"] = df["id"].str.strip()
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip().str.lower()
    df["dia"] = df["id"].str.split("_").str[0]
    df["utt"] = df["id"].str.split("_").str[1]
    df = df[df["label"] != ""].reset_index(drop=True)
    return df


def build_label2id(dfs):
    """Deterministic label->id from the actual TSV label strings (sorted)."""
    labels = sorted({l for df in dfs for l in df["label"].unique()})
    return {l: i for i, l in enumerate(labels)}


def build_stem2path():
    mp4_paths = list(VIDEO_DIR.rglob("*.mp4"))
    return {p.stem: p for p in mp4_paths}, len(mp4_paths)


def find_video(row, stem2path):
    for key in (f"MIntRec2.0_{row['id']}", row["id"], f"dia{row['dia']}_utt{row['utt']}"):
        if key in stem2path:
            return stem2path[key]
    return None


def load_teacher(dtype="bf16"):
    """dtype in {bf16, 4bit}. bf16 = full precision (cleanest KD target - matches
    the teacher's training dtype, needs ~7GB VRAM); 4bit = bnb NF4 (low-VRAM
    fallback, e.g. a 6GB laptop GPU)."""
    # sdpa (not eager): mathematically equivalent softmax attention, but fused -
    # it never materializes the full [heads, seq, seq] fp32 score matrix, so memory
    # is ~linear instead of quadratic in seq_len. hidden_states are unchanged.
    # (Long MIntRec clips = many vision+audio tokens; eager OOMs on a 22GB GPU.)
    model_kwargs = dict(device_map="auto", attn_implementation="sdpa")
    if dtype == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
    model.eval()
    return model, processor


def get_special_id(model, name, default):
    cfg = model.config
    if hasattr(cfg, name):
        return getattr(cfg, name)
    for sub in ("thinker_config", "talker_config"):
        if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), name):
            return getattr(getattr(cfg, sub), name)
    return default


def find_audio_indices(input_ids, start_id, end_id):
    ids = input_ids[0].tolist()
    s = ids.index(start_id)
    e = ids.index(end_id)
    audio_idx = list(range(s + 1, e))
    if not audio_idx:
        raise ValueError("No audio tokens found between audio start/end markers.")
    return audio_idx, s, e


def _to_cpu_fp16(t):
    return t.detach().float().cpu().to(torch.float16)


def pool_audio_mean(hidden_states, audio_idx):
    """audio_mean over the audio-token block, per layer + a mean-over-layers combo."""
    feats = {}
    per_layer = []
    for L in LLM_LAYERS:
        v = hidden_states[L][0][audio_idx].mean(dim=0)       # [hidden_dim]
        feats[f"pf_audio_mean_l{L}"] = _to_cpu_fp16(v)
        per_layer.append(v)
    combo = torch.stack(per_layer, dim=0).mean(dim=0)
    feats[f"pf_audio_mean_{MEAN_COMBO}"] = _to_cpu_fp16(combo)
    return feats


def build_inputs(processor, model, transcript, video_path, wav):
    """prompt_first, content order = [text, video(frames-only), audio]; audio LAST."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text",  "text": TASK_PROMPT_TEMPLATE.format(text=transcript)},
            {"type": "video", "video": str(video_path)},
            {"type": "audio", "audio": wav},
        ],
    }]
    text_input = processor.apply_chat_template(
        conversation, add_generation_prompt=ADD_GEN_PROMPT, tokenize=False
    )
    # use_audio_in_video=False -> video contributes frames only; audio is the separate part
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text_input, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    ).to(model.device)
    return inputs


@torch.no_grad()
def extract_sample(model, processor, transcript, video_path, wav, audio_start_id, audio_end_id):
    inputs = build_inputs(processor, model, transcript, video_path, wav)
    out = model.thinker(**inputs, output_hidden_states=True, return_dict=True)
    audio_idx, s, e = find_audio_indices(inputs["input_ids"], audio_start_id, audio_end_id)
    feats = pool_audio_mean(out.hidden_states, audio_idx)
    meta = {
        "seq_len": int(inputs["input_ids"].shape[1]),
        "num_audio_tokens": len(audio_idx),
        "audio_range": [s + 1, e],
    }
    return feats, meta


def _flush_shard(shard_dir, shard_idx, feat_bank, labels, sample_ids, metadata):
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


def _load_shards(shard_dir):
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No shards found in {shard_dir}")
    feat_bank, labels, sample_ids, metadata = {}, [], [], []
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
    done = 0
    for sp in sorted(shard_dir.glob("shard_*.pt")):
        done += len(torch.load(sp, weights_only=False)["sample_ids"])
    return done


def process_split(model, processor, df, stem2path, label2id, audio_start_id, audio_end_id,
                  shard_dir, limit=None):
    n = len(df) if limit is None else min(limit, len(df))
    shard_dir.mkdir(parents=True, exist_ok=True)

    done = _count_done(shard_dir)
    if done >= n:
        print(f"  resume: {done} samples already extracted (>= {n}) — skipping to merge.")
        return _load_shards(shard_dir)
    if done > 0:
        print(f"  resume: {done} already extracted — continuing from index {done}.")

    next_shard_idx = len(list(shard_dir.glob("shard_*.pt")))
    feat_bank, labels, sample_ids, metadata = {}, [], [], []

    for i in tqdm(range(done, n), desc="extracting", initial=done, total=n):
        row = df.iloc[i]
        vpath = find_video(row, stem2path)
        if vpath is None:
            continue
        wav, _sr = librosa.load(str(vpath), sr=16000, mono=True)
        if wav.size == 0:
            continue

        feats, meta = extract_sample(
            model, processor, row["text"], vpath, wav, audio_start_id, audio_end_id
        )
        for k, v in feats.items():
            feat_bank.setdefault(k, []).append(v)
        labels.append(label2id[row["label"]])
        sample_ids.append(row["id"])
        meta.update({"id": row["id"], "label": row["label"], "label_id": label2id[row["label"]]})
        metadata.append(meta)

        if (i + 1) % EMPTY_CACHE_EVERY == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if len(labels) >= SHARD_SIZE:
            _flush_shard(shard_dir, next_shard_idx, feat_bank, labels, sample_ids, metadata)
            next_shard_idx += 1
            feat_bank, labels, sample_ids, metadata = {}, [], [], []

    if labels:
        _flush_shard(shard_dir, next_shard_idx, feat_bank, labels, sample_ids, metadata)

    return _load_shards(shard_dir)


SPLITS = {"train": "train_features.pt", "dev": "dev_features.pt"}


def main():
    ap = argparse.ArgumentParser(description="Extract frozen multimodal-teacher audio features for MIntRec2.0.")
    ap.add_argument("--split", choices=["train", "dev", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N samples (smoke test).")
    ap.add_argument("--dtype", choices=["bf16", "4bit"], default="bf16",
                    help="Teacher precision. bf16 = full precision (RunPod, cleanest target, "
                         "matches the teacher's training dtype); 4bit = bnb NF4 (low-VRAM laptop fallback).")
    args = ap.parse_args()

    out_dir = MINTREC_DATA / "teacher_features" / build_feat_tag(args.dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem2path, n_mp4 = build_stem2path()
    print(f"Indexed {n_mp4:,} video clips under {VIDEO_DIR}")

    print(f"Loading teacher: {MODEL_NAME} ({args.dtype}) ...")
    model, processor = load_teacher(args.dtype)
    audio_start_id = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    audio_end_id   = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)
    print(f"audio_start_token_id={audio_start_id}  audio_end_token_id={audio_end_id}")

    # Shared, deterministic label map built from BOTH splits' TSVs (naming-robust).
    dfs = {s: load_split_df(s) for s in ("train", "dev")}
    label2id = build_label2id(dfs.values())
    print(f"Found {len(label2id)} intent classes: {list(label2id)}")

    targets = ["train", "dev"] if args.split == "all" else [args.split]
    for split in targets:
        df = dfs[split]
        print(f"\n=== {split}: {len(df)} in-scope samples ===")
        shard_dir = out_dir / f"{split}_shards"
        result = process_split(model, processor, df, stem2path, label2id,
                               audio_start_id, audio_end_id, shard_dir, limit=args.limit)
        out_path = out_dir / SPLITS[split]
        torch.save(result, out_path)
        print(f"Saved {len(result['labels'])} samples x {len(result['features'])} features "
              f"(dim={result['feature_dim']}) -> {out_path}")

    config = {
        "model_name": MODEL_NAME,
        "precision": args.dtype,
        "quantization": ("4bit-nf4 (bnb, double-quant, fp16 compute)"
                         if args.dtype == "4bit" else f"full ({args.dtype})"),
        "attn_implementation": "sdpa",
        "input_order": "prompt_first: [text(prompt+transcript), video(frames-only), audio]  (audio LAST)",
        "use_audio_in_video": False,
        "pooling": "audio_mean over audio-token block",
        "llm_layers": LLM_LAYERS,
        "mean_combo": f"pf_audio_mean_{MEAN_COMBO}",
        "add_generation_prompt": ADD_GEN_PROMPT,
        "task_prompt_template": TASK_PROMPT_TEMPLATE,
        "audio_start_token_id": int(audio_start_id),
        "audio_end_token_id": int(audio_end_id),
        "n_classes": len(label2id),
        "label2id": label2id,
        "feature_dtype": "float16",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_dir / 'extraction_config.json'}\nDone.")


if __name__ == "__main__":
    main()
