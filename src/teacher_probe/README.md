## Teacher Probe Experiments

We have already extracted full-scale Qwen hidden features:

* Train set: `23132` samples
* Validation set: `3118` samples
* Feature type: `prompt_first_audio_token_mean`
* Feature dimension: `2048`
* Number of classes: `31`

Important note:

* We do **not** have a separate test set.
* The original FSC validation split is being used as our **final evaluation set**.
* We do not want to tune hyperparameters on this evaluation set.
* Therefore, do **not** use validation-based early stopping or model selection based on the evaluation set.
* Train each model using a fixed training schedule and report performance only once on the evaluation set after training.

### Training setup

* Input: extracted Qwen mean vector, shape `[N, 2048]`
* Target: intent label, `31` classes
* Loss: cross entropy
* Optimizer: AdamW
* Learning rate: `1e-3` or `5e-4`
* Weight decay: `1e-4`
* Batch size: `256` or `512`
* Epochs: fixed `50`
* No early stopping
* Dropout: `0.1`
* Use train-set mean/std to standardize both train and evaluation features
* Report evaluation accuracy and macro F1 for each model
* Save the final checkpoint for each model

### Probe architectures to test

| ID | Architecture         | Bottleneck dim | Purpose                    |
| -- | -------------------- | -------------- | -------------------------- |
| A1 | `2048 -> 31`         | None           | Linear probe baseline      |
| A2 | `2048 -> 1024 -> 31` | None           | Nonlinear upper bound      |
| A3 | `2048 -> 2048 -> 31` | None           | Full-dim MLP upper bound   |
| B1 | `2048 -> 32 -> 31`   | 32             | Compact bottleneck         |
| B2 | `2048 -> 64 -> 31`   | 64             | Likely sweet spot          |
| B3 | `2048 -> 128 -> 31`  | 128            | Stable compact teacher     |
| B4 | `2048 -> 256 -> 31`  | 256            | Higher-capacity bottleneck |

For MLP layers, use:

```python
Linear
LayerNorm
GELU
Dropout(0.1)
```

For example, the bottleneck probe should be:

```text
2048 -> bottleneck_dim -> 31
```

with:

```text
Linear(2048, d)
LayerNorm(d)
GELU
Dropout(0.1)
Linear(d, 31)
```

### Optional second-round experiments

Only run these if the first-round results suggest that direct compression loses too much performance:

| ID | Architecture               | Bottleneck dim | Purpose            |
| -- | -------------------------- | -------------- | ------------------ |
| C1 | `2048 -> 512 -> 32 -> 31`  | 32             | Deeper compression |
| C2 | `2048 -> 512 -> 64 -> 31`  | 64             | Deeper compression |
| C3 | `2048 -> 512 -> 128 -> 31` | 128            | Deeper compression |

### Output

Please produce a result table like this:

| ID | Architecture | Bottleneck dim | Eval Acc | Eval Macro F1 |
| -- | ------------ | -------------- | -------- | ------------- |

Also save the bottleneck representations from the trained model if applicable, because they may be used later as teacher representations for student distillation.
