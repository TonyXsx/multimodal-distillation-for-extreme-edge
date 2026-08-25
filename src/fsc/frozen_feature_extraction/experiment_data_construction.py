"""
Builds the small stratified FSC subsets used for the layer ablation.

  train  20 per class -> 620
  val    10 per class -> 310

Test is deliberately left alone, it's reserved for the final student eval.

writes data/fsc_small_ablation/{train_20pc,val_10pc}/ as HF datasets with the
audio kept as raw bytes, plus config.json with the seed, counts and label2id.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Audio, load_dataset


import sys
_SRC = next(p for p in Path(__file__).resolve().parents if p.name == "src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.config import DATA_ROOT   # noqa: E402

SAVE_DIR  = DATA_ROOT / "fsc_small_ablation"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


SEED            = 42
TRAIN_PER_CLASS = 20
VAL_PER_CLASS   = 10


print("Loading FSC dataset...")
fsc = load_dataset("s3prl/superb", name="ic", cache_dir=str(DATA_ROOT))

features       = fsc["train"].features
action_names   = features["action"].names
object_names   = features["object"].names
location_names = features["location"].names

# decode=False keeps the audio as raw bytes and avoids the torchcodec/ffmpeg
# version mess. everything downstream decodes with soundfile instead.
fsc = fsc.cast_column("audio", Audio(sampling_rate=16000, decode=False))


def add_meta(example):
    a = action_names[example["action"]]
    o = object_names[example["object"]]
    l = location_names[example["location"]]
    example["intent"] = f"{a}_{o}_{l}"
    return example


fsc = fsc.map(add_meta, desc="Adding intent strings")

intent_labels = sorted(set(fsc["train"]["intent"]))
label2id      = {label: i for i, label in enumerate(intent_labels)}


def add_label_id(example):
    example["label_id"] = label2id[example["intent"]]
    return example


fsc = fsc.map(add_label_id, desc="Adding label IDs")


rng = random.Random(SEED)


def stratified_sample(dataset, n_per_class: int) -> list[int]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, intent in enumerate(dataset["intent"]):
        by_class[intent].append(i)

    selected: list[int] = []
    for label in intent_labels:
        pool = by_class[label]
        k = min(n_per_class, len(pool))
        if k < n_per_class:
            print(f"  WARNING: '{label}' has only {len(pool)} samples (wanted {n_per_class})")
        selected.extend(rng.sample(pool, k))

    return sorted(selected)


print(f"\nSampling train: {TRAIN_PER_CLASS} per class from FSC train split...")
train_indices = stratified_sample(fsc["train"], TRAIN_PER_CLASS)

print(f"Sampling val  : {VAL_PER_CLASS} per class from FSC validation split...")
val_indices = stratified_sample(fsc["validation"], VAL_PER_CLASS)

train_subset = fsc["train"].select(train_indices)
val_subset   = fsc["validation"].select(val_indices)


train_path = SAVE_DIR / f"train_{TRAIN_PER_CLASS}pc"
val_path   = SAVE_DIR / f"val_{VAL_PER_CLASS}pc"

print(f"\nSaving train subset -> {train_path}")
train_subset.save_to_disk(str(train_path))

print(f"Saving val subset   -> {val_path}")
val_subset.save_to_disk(str(val_path))


config = {
    "source_dataset": "s3prl/superb (ic)",
    "seed": SEED,
    "train_per_class": TRAIN_PER_CLASS,
    "val_per_class": VAL_PER_CLASS,
    "n_classes": len(intent_labels),
    "train_total": len(train_subset),
    "val_total": len(val_subset),
    "train_source_split": "train",
    "val_source_split": "validation",
    "intent_labels": intent_labels,
    "label2id": label2id,
}
config_path = SAVE_DIR / "config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)


train_counts = Counter(train_subset["intent"])
val_counts   = Counter(val_subset["intent"])

print("\n── Summary ───────────────────────────────────────────────────────────────")
print(f"  Train : {len(train_subset)} samples, {len(train_counts)} classes")
print(f"          per-class counts  min={min(train_counts.values())}  max={max(train_counts.values())}")
print(f"  Val   : {len(val_subset)} samples, {len(val_counts)} classes")
print(f"          per-class counts  min={min(val_counts.values())}  max={max(val_counts.values())}")
print(f"  Config: {config_path}")
print("\nDone. FSC test split is untouched (reserved for final evaluation).")
