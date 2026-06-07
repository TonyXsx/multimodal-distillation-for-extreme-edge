# Additional Focused Ablation Experiments

We will add only two focused ablation settings. The goal is not to run a large grid, but to test whether KD is more useful when the training data or student capacity is limited.

## Existing Main Setting

We already have the main setting:

```text
Strong student + 100% training data
```

Methods:

```text
CE-only
Logit-KD
Feature-KD
Full-KD
```

Metrics:

```text
Validation Accuracy
Macro F1
Weighted F1
```

The "strong student" recipe is not just the architecture/channels below — it also
includes the training regularization that produced the 88% baseline:

```text
- gentler frequency downsampling (blocks 3-4 stride (2,1), freq kept at 8 bins)
- BatchNorm on the projection head and the 64-dim bottleneck
- SpecAugment (time/freq masking) + label smoothing 0.1
- 70 epochs, AdamW(lr=1e-3, wd=1e-4), cosine schedule, dropout 0.2
```

The two new settings below should use the same evaluation metrics and the same
validation set, and the same training recipe (only the studied variable changes).

---

## Setting 1: Strong Student with 20% Training Data

Use the current strong DSResNet-SE student, but train it using only 20% of the original training set.

### Purpose

Test whether KD becomes more useful when labelled training data is limited.

### Model

Use the same strong student as the current 88% baseline:

```text
Strong DSResNet-SE student
channels = [32, 64, 128, 192, 256]
projection = 256 -> 128 -> 64
classifier = 64 -> 31
```

### Data

Use only 20% of the training set.

Important:

```text
Use stratified sampling if possible.
Keep the original validation set unchanged.
Do not sample from the validation set.
Use the same teacher logits and teacher bottleneck features.
```

Because 20% of the data (~4.6k samples) makes results sensitive to which subset
is drawn, run this setting with 3 different seeds (each seed = a different
stratified 20% subset + different init) and report mean +/- std per method.
This separates a real KD effect from sampling luck.

### Methods to Run

Run the following four methods:

```text
1. CE-only
2. Logit-KD
3. Feature-KD
4. Full-KD
```

### Expected Comparison

Compare these results against the existing full-data strong student setting.

Main question:

```text
Does KD, especially Feature-KD or Full-KD, provide larger gains when training data is limited?
```

---

## Setting 2: Small Student with 100% Training Data

Use the full training set, but replace the current strong student with a smaller capacity-limited DSResNet-SE student.

### Purpose

Test whether feature-level KD becomes more useful when the student model has lower capacity.

### Model

Current strong student:

```text
channels = [32, 64, 128, 192, 256]
projection = 256 -> 128 -> 64
classifier = 64 -> 31
```

New small student:

```text
channels = [16, 32, 64, 96, 128]
projection = 128 -> 64
classifier = 64 -> 31
```

Important:

```text
Keep the final projected feature dimension as 64.
This keeps the Feature-KD target dimension unchanged.
The teacher bottleneck feature is still 64-dimensional.
```

### Data

Use 100% of the original training set.

```text
Use the same train/validation split as the main setting.
Keep the validation set unchanged.
Use the same teacher logits and teacher bottleneck features.
```

### Methods to Run

Run the following four methods:

```text
1. CE-only
2. Logit-KD
3. Feature-KD
4. Full-KD
```

### Expected Comparison

Compare these results against the existing strong student full-data setting.

Main question:

```text
Does Feature-KD provide larger gains when the student has lower capacity?
```

---

## Reporting Format

Please report three tables in total:

```text
Table 1: Strong student + 100% training data
Table 2: Strong student + 20% training data
Table 3: Small student + 100% training data
```

Each table should contain:

```text
Method | Val Acc | Macro F1 | Weighted F1
```

Table 2 (20% data) reports mean +/- std over the 3 seeds.

Methods in each table:

```text
CE-only
Logit-KD
Feature-KD
Full-KD
```

---

## Notes

Do not use the test set during these ablations. The test set should remain untouched until the final model selection is complete.

For now, use the same KD hyperparameters as the current full-data experiments. The purpose is to isolate the effect of data limitation and student capacity limitation, not to run a large hyperparameter search.
