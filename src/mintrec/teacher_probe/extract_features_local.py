"""
LOCAL frozen-teacher hidden-feature extraction for MIntRec2.0 — parametrized.

Same teacher / pooling as the production extract_features.py, but built to run on a
small local GPU (tested: 6.4 GB RTX 3060, 4-bit, ~2.3 s/sample) and to sweep the
inputs we want to compare:

  --dtype       4bit (local default) | bf16
  --frames N    sub-sample N frames per clip (even; 0 = full clip, the old behaviour)
  --modalities  tva (text+video+audio)  | ta (text+audio, NO video)
  --prompt      plain                   | aware  (tells the teacher to attend to tone/face)

Input layout (prompt_first, audio LAST so its tokens absorb the preceding context):
    tva: [ text(prompt+transcript) ] + [ video frames ] + [ audio ]
    ta : [ text(prompt+transcript) ]                     + [ audio ]
use_audio_in_video=False; pool audio_mean over the audio-token block at layers
[24,27,30,34] (+ mean). Sharded + resume-safe, like the production extractor.

Every knob is encoded in FEAT_TAG so runs never collide, e.g.
    mintrec2.0__qwen2.5-omni-3b-4bit__pf_text-video4f-audio__aware__audiomean
    mintrec2.0__qwen2.5-omni-3b-4bit__pf_text-audio__plain__audiomean
All outputs land under data/mintrec/teacher_features/<FEAT_TAG>/ (data/ -> E:).

Usage:
    # smoke (5 samples), default 4bit / 4 frames / tva / plain
    python src/mintrec/teacher_probe/extract_features_local.py --split dev --limit 5
    # the requested variants:
    python .../extract_features_local.py --split all --prompt aware                 # aware prompt, T+V+A
    python .../extract_features_local.py --split all --modalities ta                # audio+transcript only
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    for _p in os.environ.get("PATH", "").split(";"):
        if _p and os.path.exists(os.path.join(_p, "avcodec-62.dll")):
            os.add_dll_directory(_p)
            break

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
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
from common.config import MINTREC_DATA  # noqa: E402

ANNO_DIR  = MINTREC_DATA / "MIntRec2.0"
VIDEO_DIR = ANNO_DIR / "video"

MODEL_NAME   = "Qwen/Qwen2.5-Omni-3B"
LLM_LAYERS   = [24, 27, 30, 34]
MEAN_COMBO   = "L24-27-30-34"
ADD_GEN_PROMPT    = True
FRAME_MAX_SIDE    = 336
AUDIO_SR          = 16000
SHARD_SIZE        = 50
EMPTY_CACHE_EVERY = 5
AUDIO_START_ID_DEFAULT = 151647
AUDIO_END_ID_DEFAULT   = 151648

# ── prompts ─────────────────────────────────────────────────────────────────────────
_BASE = ("You are analyzing a short TV-show clip to recognize the speaker's intent "
         "among 30 fine-grained intent classes. ")
_AWARE_VID = ("Pay close attention to HOW it is said — the speaker's tone of voice, "
              "prosody, and facial expression — not just the words. ")
_AWARE_AUD = ("Pay close attention to HOW it is said — the speaker's tone of voice "
              "and prosody — not just the words. ")
_TRANSCRIPT = 'The transcript of what is said is: "{text}".'


def build_prompt(style, has_video, text):
    p = _BASE
    if style == "aware":
        p += (_AWARE_VID if has_video else _AWARE_AUD)
    return p + _TRANSCRIPT.format(text=text)


def build_feat_tag(dtype, modalities, frames, prompt):
    if modalities == "tva":
        modtag = f"pf_text-video{frames}f-audio" if frames else "pf_text-video-audio"
    else:
        modtag = "pf_text-audio"
    return f"mintrec2.0__qwen2.5-omni-3b-{dtype}__{modtag}__{prompt}__audiomean"


# ── annotations (mirrors production extractor) ───────────────────────────────────────
def load_split_df(name):
    df = pd.read_csv(ANNO_DIR / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["id"] = df["id"].str.strip()
    df["text"] = df["text"].str.strip()
    df["label"] = df["label"].str.strip().str.lower()
    df["dia"] = df["id"].str.split("_").str[0]
    df["utt"] = df["id"].str.split("_").str[1]
    return df[df["label"] != ""].reset_index(drop=True)


def build_label2id(dfs):
    labels = sorted({l for df in dfs for l in df["label"].unique()})
    return {l: i for i, l in enumerate(labels)}


def build_stem2path():
    mp4 = list(VIDEO_DIR.rglob("*.mp4"))
    return {p.stem: p for p in mp4}, len(mp4)


def find_video(row, s2p):
    for k in (f"MIntRec2.0_{row['id']}", row["id"], f"dia{row['dia']}_utt{row['utt']}"):
        if k in s2p:
            return s2p[k]
    return None


# ── media (cv2 frame sampling + audio from mp4) ──────────────────────────────────────
def _resize_max_side(img, ms):
    h, w = img.shape[:2]
    sc = ms / max(h, w)
    if sc < 1.0:
        img = cv2.resize(img, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
    return img


def extract_frames(path, n_frames, max_side=FRAME_MAX_SIDE):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    end = (total / fps) if total else 1.0
    times = np.linspace(0.0, end, n_frames + 2)[1:-1]
    frames = []
    for t in times:
        idx = min(int(round(t * fps)), total - 1) if total else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(_resize_max_side(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), max_side)))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames from {path}")
    if len(frames) % 2 == 1:                       # Qwen temporal patch = 2 -> even count
        frames = frames[:-1] if len(frames) > 1 else frames + frames
    return frames


def load_audio(path):
    wav, _ = librosa.load(str(path), sr=AUDIO_SR, mono=True)
    return wav


# ── model ─────────────────────────────────────────────────────────────────────────
def load_teacher(dtype):
    kw = dict(device_map="auto", attn_implementation="sdpa")
    if dtype == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    else:
        kw["torch_dtype"] = torch.bfloat16
    proc = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
    mdl = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_NAME, **kw)
    mdl.eval()
    return mdl, proc


def get_special_id(model, name, default):
    cfg = model.config
    if hasattr(cfg, name):
        return getattr(cfg, name)
    for sub in ("thinker_config", "talker_config"):
        if hasattr(cfg, sub) and hasattr(getattr(cfg, sub), name):
            return getattr(getattr(cfg, sub), name)
    return default


def find_audio_indices(input_ids, s_id, e_id):
    ids = input_ids[0].tolist()
    s, e = ids.index(s_id), ids.index(e_id)
    idx = list(range(s + 1, e))
    if not idx:
        raise ValueError("no audio tokens between start/end markers")
    return idx, s, e


def _cpu16(t):
    return t.detach().float().cpu().to(torch.float16)


def pool_audio_mean(hs, audio_idx):
    feats, per = {}, []
    for L in LLM_LAYERS:
        v = hs[L][0][audio_idx].mean(0)
        feats[f"pf_audio_mean_l{L}"] = _cpu16(v)
        per.append(v)
    feats[f"pf_audio_mean_{MEAN_COMBO}"] = _cpu16(torch.stack(per, 0).mean(0))
    return feats


def build_inputs(proc, model, text, frames, wav):
    content = [{"type": "text", "text": text}]
    if frames is not None:
        content.append({"type": "video", "video": frames})
    content.append({"type": "audio", "audio": wav})
    conv = [{"role": "user", "content": content}]
    txt = proc.apply_chat_template(conv, add_generation_prompt=ADD_GEN_PROMPT, tokenize=False)
    a, i, v = process_mm_info(conv, use_audio_in_video=False)
    return proc(text=txt, audio=a, images=i, videos=v, return_tensors="pt",
                padding=True, use_audio_in_video=False).to(model.device)


@torch.no_grad()
def extract_sample(model, proc, text, frames, wav, a0, a1):
    inp = build_inputs(proc, model, text, frames, wav)
    out = model.thinker(**inp, output_hidden_states=True, return_dict=True)
    audio_idx, s, e = find_audio_indices(inp["input_ids"], a0, a1)
    feats = pool_audio_mean(out.hidden_states, audio_idx)
    return feats, {"seq_len": int(inp["input_ids"].shape[1]), "num_audio_tokens": len(audio_idx)}


# ── sharding (ported from production extractor) ──────────────────────────────────────
def _flush_shard(shard_dir, idx, fb, labels, ids, meta):
    shard = {"labels": torch.tensor(labels, dtype=torch.long), "sample_ids": ids, "metadata": meta,
             "features": {k: torch.stack(v, 0) for k, v in fb.items()}}
    tmp = shard_dir / f"shard_{idx:05d}.pt.tmp"
    torch.save(shard, tmp)
    os.replace(tmp, shard_dir / f"shard_{idx:05d}.pt")


def _load_shards(shard_dir):
    paths = sorted(shard_dir.glob("shard_*.pt"))
    fb, labels, ids, meta = {}, [], [], []
    for sp in paths:
        sh = torch.load(sp, weights_only=False)
        labels.append(sh["labels"]); ids.extend(sh["sample_ids"]); meta.extend(sh["metadata"])
        for k, v in sh["features"].items():
            fb.setdefault(k, []).append(v)
    features = {k: torch.cat(v, 0) for k, v in fb.items()}
    return {"labels": torch.cat(labels, 0), "sample_ids": ids, "metadata": meta,
            "feature_dim": next(iter(features.values())).shape[1], "features": features}


def _count_done(shard_dir):
    return sum(len(torch.load(sp, weights_only=False)["sample_ids"])
               for sp in sorted(shard_dir.glob("shard_*.pt")))


def process_split(model, proc, df, s2p, label2id, a0, a1, shard_dir, args):
    n = len(df) if args.limit is None else min(args.limit, len(df))
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = _count_done(shard_dir)
    if done >= n:
        print(f"  resume: {done} already extracted (>= {n}) — merging.")
        return _load_shards(shard_dir)
    if done > 0:
        print(f"  resume: {done} already extracted — continuing.")

    next_idx = len(list(shard_dir.glob("shard_*.pt")))
    fb, labels, ids, meta = {}, [], [], []
    has_video = (args.modalities == "tva")

    for i in tqdm(range(done, n), desc="extract", initial=done, total=n):
        row = df.iloc[i]
        vp = find_video(row, s2p)
        if vp is None:
            continue
        try:
            wav = load_audio(vp)
            if wav.size == 0:
                continue
            frames = extract_frames(vp, args.frames) if has_video else None
            text = build_prompt(args.prompt, has_video, row["text"])
            feats, m = extract_sample(model, proc, text, frames, wav, a0, a1)
        except Exception as ex:
            print(f"  skip {row['id']}: {type(ex).__name__}: {ex}")
            if torch.cuda.is_available():          # recover from a transient OOM / bad clip
                torch.cuda.empty_cache(); gc.collect()
            continue
        for k, v in feats.items():
            fb.setdefault(k, []).append(v)
        labels.append(label2id[row["label"]]); ids.append(row["id"])
        m.update({"id": row["id"], "label": row["label"], "label_id": label2id[row["label"]]})
        meta.append(m)

        if (i + 1) % EMPTY_CACHE_EVERY == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()
        if len(labels) >= SHARD_SIZE:
            _flush_shard(shard_dir, next_idx, fb, labels, ids, meta)
            next_idx += 1
            fb, labels, ids, meta = {}, [], [], []

    if labels:
        _flush_shard(shard_dir, next_idx, fb, labels, ids, meta)
    return _load_shards(shard_dir)


SPLIT_FILE = {"train": "train_features.pt", "dev": "dev_features.pt", "test": "test_features.pt"}


def main():
    ap = argparse.ArgumentParser(description="Local parametrized MIntRec2.0 teacher feature extraction.")
    ap.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="first N samples (smoke test)")
    ap.add_argument("--dtype", choices=["4bit", "bf16"], default="4bit")
    ap.add_argument("--frames", type=int, default=4, help="sub-sampled frames/clip (even; 0=full clip). Ignored if --modalities ta.")
    ap.add_argument("--modalities", choices=["tva", "ta"], default="tva", help="tva=text+video+audio, ta=text+audio (no video)")
    ap.add_argument("--prompt", choices=["plain", "aware"], default="plain")
    args = ap.parse_args()
    if args.modalities == "ta":
        args.frames = 0
    elif args.frames and args.frames % 2 == 1:
        raise SystemExit("--frames must be even (Qwen temporal patch=2)")

    assert ANNO_DIR.exists(), f"raw MIntRec2.0 not found at {ANNO_DIR} — run download_data.py first"
    tag = build_feat_tag(args.dtype, args.modalities, args.frames, args.prompt)
    out_dir = MINTREC_DATA / "teacher_features" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"FEAT_TAG: {tag}\nOUT: {out_dir}")

    s2p, n_mp4 = build_stem2path()
    print(f"Indexed {n_mp4:,} clips under {VIDEO_DIR}")
    dfs = {s: load_split_df(s) for s in ("train", "dev", "test") if (ANNO_DIR / f"{s}.tsv").exists()}
    label2id = build_label2id(dfs.values())
    print(f"{len(label2id)} intent classes")

    print(f"Loading teacher ({args.dtype}) ...")
    model, proc = load_teacher(args.dtype)
    a0 = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    a1 = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)

    targets = list(dfs) if args.split == "all" else [args.split]
    for split in targets:
        print(f"\n=== {split}: {len(dfs[split])} samples ===")
        result = process_split(model, proc, dfs[split], s2p, label2id, a0, a1,
                               out_dir / f"{split}_shards", args)
        torch.save(result, out_dir / SPLIT_FILE[split])
        print(f"Saved {len(result['labels'])} x {len(result['features'])} feats "
              f"(dim={result['feature_dim']}) -> {out_dir / SPLIT_FILE[split]}")

    config = {
        "model_name": MODEL_NAME, "precision": args.dtype, "modalities": args.modalities,
        "frames": args.frames, "frame_max_side": FRAME_MAX_SIDE, "prompt_style": args.prompt,
        "prompt_example": build_prompt(args.prompt, args.modalities == "tva", "<TRANSCRIPT>"),
        "input_order": ("prompt_first [text, video-frames, audio]" if args.modalities == "tva"
                        else "prompt_first [text, audio]") + " (audio LAST)",
        "use_audio_in_video": False, "pooling": "audio_mean over audio-token block",
        "llm_layers": LLM_LAYERS, "mean_combo": f"pf_audio_mean_{MEAN_COMBO}",
        "n_classes": len(label2id), "label2id": label2id, "feature_dtype": "float16",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "extraction_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_dir / 'extraction_config.json'}\nDone.")


if __name__ == "__main__":
    main()
