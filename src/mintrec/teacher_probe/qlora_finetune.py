"""
QLoRA fine-tune Qwen2.5-Omni (Thinker) for MIntRec2.0 intent recognition.

This turns the WEAK frozen teacher (~58% dev/test from the probe) into a strong,
task-adapted teacher whose hidden states become a much better KD target. Recipe
follows the discriminative-readout MSA paper (arXiv 2606.05713): 4-bit NF4 backbone
+ LoRA on the LLM projections + a lightweight classification head on the pooled
last-token representation, one forward pass, layer-wise LRs.

INTENDED FOR RUNPOD (Linux, >=24 GB GPU) — will NOT fit the 6 GB laptop (training
needs gradients/optimizer state on top of the ~5 GB inference footprint).

After training, load the saved adapter and re-run feature extraction with LoRA
applied to get the ADAPTED hidden states as the KD target (follow-up step).

Setup (RunPod):
    pip install "transformers>=4.52" accelerate bitsandbytes peft \
                "qwen-omni-utils[decord]" librosa soundfile av opencv-python-headless scikit-learn
    # data already at data/mintrec/MIntRec2.0/ (run download_data.py if not)

Smoke then full:
    python src/mintrec/teacher_probe/qlora_finetune.py --limit 40 --epochs 1   # sanity
    python src/mintrec/teacher_probe/qlora_finetune.py --model 3b --frames 8 --epochs 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_omni_utils import process_mm_info

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import MINTREC_DATA  # noqa: E402
# reuse the data helpers from the local extractor (pure data, no model state)
from mintrec.teacher_probe.extract_features_local import (  # noqa: E402
    load_split_df, build_label2id, build_stem2path, find_video,
    extract_frames, load_audio, build_prompt, get_special_id, ANNO_DIR,
)

MODELS = {"3b": "Qwen/Qwen2.5-Omni-3B", "7b": "Qwen/Qwen2.5-Omni-7B"}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
AUDIO_START_ID_DEFAULT, AUDIO_END_ID_DEFAULT = 151647, 151648


def get_hidden_size(model):
    cfg = model.config
    for path in ("thinker_config.text_config.hidden_size",
                 "thinker_config.hidden_size", "text_config.hidden_size", "hidden_size"):
        o = cfg
        try:
            for a in path.split("."):
                o = getattr(o, a)
            if isinstance(o, int):
                return o
        except AttributeError:
            continue
    raise RuntimeError("could not infer hidden size from config")


class OmniClassifier(nn.Module):
    """LoRA-adapted Qwen Thinker + pooled last-token -> MLP classification head."""

    def __init__(self, thinker, hidden, n_classes, pool="last", head_hidden=256, dropout=0.2):
        super().__init__()
        self.thinker = thinker
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.LayerNorm(head_hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(head_hidden, n_classes),
        )

    def forward(self, inputs, audio_ids=None):
        out = self.thinker(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]                       # [B, T, H] final layer
        am = inputs["attention_mask"]
        if self.pool == "last":
            last = am.sum(1) - 1                         # last non-pad token (mask-safe)
            z = h[torch.arange(h.size(0), device=h.device), last]
        else:                                            # audio_mean (batch size 1)
            s, e = audio_ids
            z = h[0, s + 1:e].mean(0, keepdim=True)
        return self.head(z.to(self.head[0].weight.dtype))


def load_backbone(model_name, compute_dtype=torch.bfloat16):
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    proc = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name, device_map={"": 0}, attn_implementation="sdpa", quantization_config=qc)
    # free the speech-generation half — we only train/use the Thinker
    for attr in ("talker", "token2wav"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                setattr(model, attr, None)
    return model, proc


def build_inputs(proc, device, text, frames, wav):
    content = [{"type": "text", "text": text}]
    if frames is not None:
        content.append({"type": "video", "video": frames})
    content.append({"type": "audio", "audio": wav})
    conv = [{"role": "user", "content": content}]
    txt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    a, i, v = process_mm_info(conv, use_audio_in_video=False)
    return proc(text=txt, audio=a, images=i, videos=v, return_tensors="pt",
                padding=True, use_audio_in_video=False).to(device)


@torch.no_grad()
def evaluate(clf, proc, device, df, s2p, label2id, args, a0, a1):
    clf.eval()
    ys, ps = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="eval", leave=False):
        vp = find_video(row, s2p)
        if vp is None:
            continue
        try:
            wav = load_audio(vp)
            if wav.size == 0:
                continue
            frames = extract_frames(vp, args.frames) if args.modalities == "tva" else None
            text = build_prompt(args.prompt, args.modalities == "tva", row["text"])
            inp = build_inputs(proc, device, text, frames, wav)
            aud = None
            if args.pool == "audio_mean":
                ids = inp["input_ids"][0].tolist()
                aud = (ids.index(a0), ids.index(a1))
            logits = clf(inp, aud)
        except Exception as ex:
            print("  eval skip", row["id"], type(ex).__name__, ex)
            continue
        ps.append(int(logits.argmax(-1)[0])); ys.append(label2id[row["label"]])
    acc = float(np.mean(np.array(ps) == np.array(ys)))
    f1 = f1_score(ys, ps, average="macro")
    return acc, f1


def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-Omni Thinker on MIntRec2.0.")
    ap.add_argument("--model", choices=["3b", "7b"], default="3b")
    ap.add_argument("--frames", type=int, default=8, help="sub-sampled frames/clip (even)")
    ap.add_argument("--modalities", choices=["tva", "ta"], default="tva")
    ap.add_argument("--prompt", choices=["plain", "aware"], default="aware")
    ap.add_argument("--pool", choices=["last", "audio_mean"], default="last")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=16, help="grad-accumulation steps (physical batch=1)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N train samples (smoke)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.modalities == "ta":
        args.frames = 0
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    assert ANNO_DIR.exists(), f"raw MIntRec2.0 not found at {ANNO_DIR}"

    device = "cuda"
    model_name = MODELS[args.model]
    print(f"Loading {model_name} (4-bit NF4) ...")
    model, proc = load_backbone(model_name)
    hidden = get_hidden_size(model)
    a0 = get_special_id(model, "audio_start_token_id", AUDIO_START_ID_DEFAULT)
    a1 = get_special_id(model, "audio_end_token_id", AUDIO_END_ID_DEFAULT)

    thinker = prepare_model_for_kbit_training(model.thinker, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
                      bias="none", target_modules=LORA_TARGETS)
    thinker = get_peft_model(thinker, lora)
    thinker.print_trainable_parameters()
    if hasattr(thinker, "config"):
        thinker.config.use_cache = False

    # data
    s2p, n_mp4 = build_stem2path()
    dfs = {s: load_split_df(s) for s in ("train", "dev") if (ANNO_DIR / f"{s}.tsv").exists()}
    label2id = build_label2id(dfs.values())
    n_classes = len(label2id)
    train_df = dfs["train"] if args.limit is None else dfs["train"].iloc[:args.limit]
    print(f"{n_mp4} clips | {n_classes} classes | train {len(train_df)} dev {len(dfs['dev'])}")

    clf = OmniClassifier(thinker, hidden, n_classes, pool=args.pool).to(device)
    clf.head.to(torch.bfloat16)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([
        {"params": [p for p in thinker.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": clf.head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-4)
    steps = (len(train_df) * args.epochs) // args.accum
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), max(steps, 1))

    out_dir = MINTREC_DATA / "teacher_qlora" / f"{args.model}_{args.modalities}_{args.prompt}_{args.pool}_r{args.lora_r}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        clf.train(); opt.zero_grad()
        order = np.random.permutation(len(train_df))
        running = 0.0; seen = 0
        pbar = tqdm(order, desc=f"epoch {ep+1}/{args.epochs}")
        for step, ridx in enumerate(pbar):
            row = train_df.iloc[int(ridx)]
            vp = find_video(row, s2p)
            if vp is None:
                continue
            try:
                wav = load_audio(vp)
                if wav.size == 0:
                    continue
                frames = extract_frames(vp, args.frames) if args.modalities == "tva" else None
                text = build_prompt(args.prompt, args.modalities == "tva", row["text"])
                inp = build_inputs(proc, device, text, frames, wav)
                aud = None
                if args.pool == "audio_mean":
                    ids = inp["input_ids"][0].tolist(); aud = (ids.index(a0), ids.index(a1))
                logits = clf(inp, aud)
                y = torch.tensor([label2id[row["label"]]], device=device)
                loss = loss_fn(logits, y) / args.accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                print("  OOM, skip", row["id"]); opt.zero_grad(); torch.cuda.empty_cache(); continue
            except Exception as ex:
                print("  skip", row["id"], type(ex).__name__, ex); continue
            running += loss.item() * args.accum; seen += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in thinker.parameters() if p.requires_grad] + list(clf.head.parameters()), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                pbar.set_postfix(loss=f"{running/max(seen,1):.3f}")
        acc, f1 = evaluate(clf, proc, device, dfs["dev"], s2p, label2id, args, a0, a1)
        print(f"[epoch {ep+1}] train_loss={running/max(seen,1):.4f}  dev_acc={acc:.4f}  dev_macroF1={f1:.4f}")
        thinker.save_pretrained(out_dir / f"adapter_ep{ep+1}")
        torch.save(clf.head.state_dict(), out_dir / f"head_ep{ep+1}.pt")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "model_name": model_name, "hidden": hidden,
                   "n_classes": n_classes, "label2id": label2id,
                   "lora_targets": LORA_TARGETS}, f, indent=2, ensure_ascii=False)
    print(f"Saved adapters + head + config -> {out_dir}")


if __name__ == "__main__":
    main()
