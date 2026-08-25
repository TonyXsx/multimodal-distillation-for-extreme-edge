"""
Backbone loading and the LoRA config used for IEMOCAP.

Two things differ from the MIntRec teacher, both on purpose.

bf16 by default instead of 4-bit. The "3B" in Qwen2.5-Omni-3B is misleading -
the Thinker alone is 4.703 B params (LLM 3.086 + audio tower 0.638 + visual
tower 0.669 + lm_head 0.311). Full fine-tuning is out of reach on one 24 GB
card, since weights + grads + fp32 master + AdamW moments is about 16 bytes a
parameter, so ~75 GB. LoRA on a bf16 base fits fine though: 9.4 GB frozen
weights, ~1.5 GB adapter grads and optimiser state, 1-2 GB activations at batch
1 with checkpointing.

MIntRec needed 4-bit because every sample carried 8 video frames. IEMOCAP is
audio only at ~4.6 s mean duration so that memory isn't needed any more.
Dropping quantisation also gets rid of a caveat the write-up would otherwise
have to make, and skips a dequant step on every matmul, which is faster.
--dtype 4bit is still there as a small-GPU fallback, but the two are not
interchangeable - features have to be extracted at whatever precision the
adapter was trained at.

The LoRA coverage follows the task rather than MIntRec's module list. Counting
the MIntRec adapter shows where its 82.2 M params went: 59.9 M to the LLM,
14.4 M to the visual tower, and only 7.9 M to the audio tower - and that last
bit was just q/k/v, because the target list uses LLM naming (o_proj, gate_proj,
up_proj, down_proj) while the audio tower is Whisper-style and calls the same
things out_proj, fc1, fc2. Those never matched.

For a text-heavy intent task that didn't matter much. For emotion it does,
since prosody is exactly what the audio tower encodes and it was the one part
left half-adapted. So here the audio tower is covered properly (out_proj, fc1,
fc2 added), the visual tower is excluded since there's no video on this track,
and the audio tower gets a higher rank than the language backbone.
"""

import re
import sys
from pathlib import Path

import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from peft import LoraConfig

_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from mintrec.teacher_probe.qlora_finetune import get_hidden_size  # noqa: E402,F401

MODELS = {"3b": "Qwen/Qwen2.5-Omni-3B", "7b": "Qwen/Qwen2.5-Omni-7B"}

# LLM decoder uses Qwen2 naming, the audio tower uses Whisper naming. out_proj,
# fc1 and fc2 only exist in the audio tower, the rest only in the LLM
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",        # LLM attention
    "gate_proj", "up_proj", "down_proj",           # LLM MLP
    "out_proj", "fc1", "fc2",                      # audio tower attention out + MLP
]
# the visual tower matches gate_proj/up_proj/down_proj by name as well, which is
# how MIntRec leaked 14.4 M params into a branch it never fed. excluded by path
LORA_EXCLUDE = r".*visual.*"

AUDIO_PATTERN = r".*audio_tower.*"


def load_thinker(model_name, dtype="bf16", device_map=None):
    """load Qwen2.5-Omni and drop the speech-generation half.

    Returns (model, processor), caller uses model.thinker. The visual tower
    stays loaded - keeping it out of LoRA is enough, and deleting it might
    break a forward pass that still references it.
    """
    kw = {"attn_implementation": "sdpa",
          "device_map": device_map if device_map is not None else {"": 0}}
    if dtype == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif dtype == "bf16":
        kw["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"unknown dtype {dtype!r}")

    proc = Qwen2_5OmniProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **kw)
    for attr in ("talker", "token2wav"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                setattr(model, attr, None)
    return model, proc


def build_lora_config(r=32, alpha=64, dropout=0.05, audio_r=64, audio_alpha=128):
    """full LLM plus full audio tower, no visual, audio at a higher rank.

    audio_r=0 gives a uniform rank everywhere instead.
    """
    kw = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
              target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE)
    if audio_r:
        kw["rank_pattern"] = {AUDIO_PATTERN: audio_r}
        kw["alpha_pattern"] = {AUDIO_PATTERN: audio_alpha}
    return LoraConfig(**kw)


def describe_trainable(model, verbose=True):
    """trainable params broken down per component.

    Printed before training so I can see the LoRA actually landed where it was
    supposed to. The MIntRec adapter looked fine until someone counted the
    tensors.
    """
    groups, total, trainable = {}, 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if not p.requires_grad:
            continue
        trainable += p.numel()
        if "audio_tower" in name:
            comp = "audio_tower"
        elif "visual" in name:
            comp = "visual"
        elif re.search(r"model\.layers", name):
            comp = "llm.layers"
        else:
            comp = "other"
        mod = re.search(r"\.([a-z_0-9]+)\.lora_[AB]", name)
        key = (comp, mod.group(1) if mod else "-")
        groups[key] = groups.get(key, 0) + p.numel()

    if verbose:
        print(f"trainable {trainable/1e6:.1f} M / {total/1e9:.3f} B "
              f"({100*trainable/max(total,1):.3f}%)")
        for (comp, mod), n in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"    {comp:12s} {mod:12s} {n/1e6:7.2f} M")
        by_comp = {}
        for (comp, _), n in groups.items():
            by_comp[comp] = by_comp.get(comp, 0) + n
        print("  per component: " + ", ".join(
            f"{c}={n/1e6:.1f}M" for c, n in sorted(by_comp.items(), key=lambda x: -x[1])))
        if by_comp.get("visual"):
            print("  WARNING: visual tower received LoRA parameters -- exclusion failed")
        if not by_comp.get("audio_tower"):
            print("  WARNING: audio tower received NO LoRA parameters")
    return groups, trainable, total
