# Student Model Design: DSResNet-SE with KD Ablation

## Goal

We want to train a compact audio-only student model for FSC intent classification. The student should learn from the Qwen2.5-Omni teacher bottleneck representation and teacher logits. The model should be small enough for edge/TinyML-style deployment, ideally only a few MB in FP32 and much smaller after INT8 quantisation.

---

## Student Architecture

Use one fixed student architecture for all experiments:

```text
Input waveform
→ 64-bin log-mel spectrogram
→ compact DSResNet-SE audio encoder
→ global average pooling
→ 64-dim projection head
→ 31-way classifier
```

### Detailed architecture

```text
Input:
  waveform sampled at 16 kHz

Feature extraction:
  64-bin log-mel spectrogram

Encoder:
  Conv2D stem: 1 → 32 channels

  ResDS-SE block 1: 32 → 64 channels
  ResDS-SE block 2: 64 → 128 channels
  ResDS-SE block 3: 128 → 192 channels
  ResDS-SE block 4: 192 → 256 channels

Pooling:
  global average pooling over time-frequency dimensions

Projection head:
  Linear 256 → 128
  ReLU
  Dropout
  Linear 128 → 64

Classifier:
  Linear 64 → 31
```

Each `ResDS-SE block` should contain residual depthwise-separable convolution and squeeze-and-excitation channel attention:

```text
x
→ depthwise conv 3×3
→ pointwise conv 1×1
→ batch norm
→ ReLU
→ depthwise conv 3×3
→ pointwise conv 1×1
→ batch norm
→ SE block
→ residual addition
→ ReLU
```

If input and output channels are different, use a 1×1 convolution on the residual path to match dimensions.

Use SE reduction ratio `r = 8`.

Expected model size:

```text
Parameters: roughly 0.35M–0.45M
FP32 weight size: roughly 1.5–2 MB
FP16 weight size: roughly 0.7–1 MB
INT8 weight size: roughly 0.35–0.5 MB
```

The exact number can be printed after model construction.

---

## Teacher Signals

The teacher side is already trained:

```text
Qwen hidden representation, 2048 dim
→ teacher bottleneck projection, 64 dim
→ teacher classifier logits, 31 dim
```

For each training sample, load:

```text
teacher_z_64: teacher bottleneck feature, shape [B, 64]
teacher_logits: teacher classifier logits, shape [B, 31]
label: ground-truth FSC intent label, shape [B]
```

The student should output:

```text
student_z_64: student projected feature, shape [B, 64]
student_logits: student classifier logits, shape [B, 31]
```

---

## Loss Functions

### 1. Cross-entropy loss

```text
L_ce = CE(student_logits, label)
```

### 2. Logit distillation loss

Use KL divergence with temperature:

```text
L_logit = KL(
    log_softmax(student_logits / T),
    softmax(teacher_logits / T)
) * T^2
```

Default:

```text
T = 2 or 4
```

Start with `T = 2`.

### 3. Feature distillation loss

Use cosine distance between student and teacher bottleneck features:

```text
L_feature = 1 - cosine_similarity(student_z_64, teacher_z_64).mean()
```

Optionally also try MSE later, but the first version should use cosine loss.

### 4. Full loss

```text
L_total = L_ce + lambda_logit * L_logit + lambda_feature * L_feature
```

Default weights:

```text
lambda_logit = 0.5
lambda_feature = 1.0
```

---

## Ablation Experiments

Use the same DSResNet-SE architecture for all experiments. Only change the training loss.

| Experiment       | Loss                                                         |
| ---------------- | ------------------------------------------------------------ |
| CE-only baseline | `L_ce`                                                       |
| Logit-KD only    | `L_ce + lambda_logit * L_logit`                              |
| Feature-KD only  | `L_ce + lambda_feature * L_feature`                          |
| Full KD          | `L_ce + lambda_logit * L_logit + lambda_feature * L_feature` |

Recommended default setting:

```text
T = 2
lambda_logit = 0.5
lambda_feature = 1.0
```

---

## Evaluation

Evaluate every model on the FSC validation set.

Report:

```text
validation accuracy
macro-F1
weighted-F1
parameter count
FP32 model size in MB
optional: INT8 estimated size
```

The key comparison is not architecture comparison. The key comparison is whether Qwen teacher distillation improves the same compact student:

```text
CE-only
vs Logit-KD
vs Feature-KD
vs Full-KD
```

Expected interpretation:

```text
If Feature-KD improves over CE-only, this supports the idea that the 64-dim Qwen bottleneck contains useful task-level semantic information.

If Full-KD performs best, this supports combining class-level soft targets and representation-level semantic alignment.

If CE-only is already very strong, the FSC task may be too easy, but KD can still be useful if it improves convergence, macro-F1, or low-capacity performance.
```

---

## Implementation Notes

Please implement the model cleanly in PyTorch.

Requirements:

```text
1. The model forward should return both student_z_64 and student_logits.
2. Print total trainable parameters.
3. Estimate FP32 model size as params * 4 / 1024^2 MB.
4. Keep the architecture fixed across all ablations.
5. Use the same train/val split and same random seed for all experiments.
6. Save the best checkpoint based on validation macro-F1 or validation accuracy.
7. Log train loss, validation accuracy, and validation macro-F1 for each epoch.
```

Forward output format:

```python
student_z, student_logits = model(waveform_or_logmel)
```

If log-mel features are precomputed, the model can directly take log-mel input with shape:

```text
[B, 1, T, 64]
```

Otherwise, implement log-mel extraction in the dataset or collate pipeline.
