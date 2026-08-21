"""
Backbone loading and the IEMOCAP-tuned LoRA configuration.

Two things differ from the MIntRec teacher, both deliberate.

PRECISION -- bf16 by default, not 4-bit.

    "Qwen2.5-Omni-3B" is a misleading name: the Thinker alone is 4.703 B
    parameters (LLM backbone 3.086 B + audio tower 0.638 B + visual tower
    0.669 B + lm_head 0.311 B). Full fine-tuning it is out of reach on one
    24 GB card -- weights + grads + fp32 master + AdamW moments is ~16 bytes
    per parameter, i.e. ~75 GB. LoRA on a bf16 base, however, fits with room
    to spare: 9.4 GB frozen weights + ~1.5 GB for the adapter's grads and
    optimiser state + 1-2 GB of activations at batch 1 with checkpointing.

    MIntRec used 4-bit because each of its samples carried eight video frames;
    IEMOCAP is audio-only at ~4.6 s mean duration, so the memory that bought
    is no longer needed. Dropping quantisation also removes a caveat the
    thesis currently has to state -- that the teacher is 4-bit and therefore
    conservative -- and avoids a dequantisation step on every matmul, which
    is simply faster. `--dtype 4bit` remains available for a small-GPU
    fallback, but the two are NOT interchangeable: features must be extracted
    at the precision the adapter was trained at.

LoRA COVERAGE -- follows the task, not MIntRec's module list.

    Inspecting the MIntRec adapter shows where its 82.2 M parameters actually
    went: 59.9 M into the LLM, 14.4 M into the VISUAL tower, and only 7.9 M
    into the audio tower -- and that last part covered just q/k/v, because
    MIntRec's target list uses LLM naming (`o_proj`, `gate_proj`, `up_proj`,
    `down_proj`) while the audio tower is Whisper-style and names the same
    roles `out_proj`, `fc1`, `fc2`. Nothing matched them.

    For a text-dominated intent task that hardly mattered. For emotion it
    does: prosody is what the audio tower encodes, and it was the one
    component left half-adapted. So here the audio tower is covered fully
    (`out_proj`, `fc1`, `fc2` added), the visual tower is excluded outright
    since this track has no video, and the audio tower is additionally given
    a HIGHER rank than the language backbone -- capacity placed where the
    task's signal lives rather than spread uniformly.
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

# LLM decoder (Qwen2 naming) + audio tower (Whisper naming). `out_proj`, `fc1`
# and `fc2` exist only in the audio tower; the rest only in the LLM.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",        # LLM attention
    "gate_proj", "up_proj", "down_proj",           # LLM MLP
    "out_proj", "fc1", "fc2",                      # audio tower attention out + MLP
]
# The visual tower reaches `gate_proj`/`up_proj`/`down_proj` by name too, which
# is exactly how MIntRec leaked 14.4 M parameters into a branch this track
# never feeds. Excluded by path.
LORA_EXCLUDE = r".*visual.*"

AUDIO_PATTERN = r".*audio_tower.*"


def load_thinker(model_name, dtype="bf16", device_map=None):
    """Load Qwen2.5-Omni and drop the speech-generation half (Thinker only).

    Returns (model, processor); the caller uses `model.thinker`. The visual
    tower is left loaded -- excluding it from LoRA is enough, and deleting it
    risks breaking a forward pass that may still reference it.
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
    """IEMOCAP LoRA: full LLM + full audio tower, no visual, audio at higher rank.

    `audio_r=0` falls back to a uniform rank everywhere.
    """
    kw = dict(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
              target_modules=LORA_TARGETS, exclude_modules=LORA_EXCLUDE)
    if audio_r:
        kw["rank_pattern"] = {AUDIO_PATTERN: audio_r}
        kw["alpha_pattern"] = {AUDIO_PATTERN: audio_alpha}
    return LoraConfig(**kw)


def describe_trainable(model, verbose=True):
    """Per-component trainable-parameter breakdown.

    Printed before training starts so the LoRA actually landed where intended
    is verified rather than assumed -- the MIntRec adapter looked fine until
    its tensors were counted.
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
