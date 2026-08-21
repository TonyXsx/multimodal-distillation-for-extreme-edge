# MS-SENet as a student backbone: evaluated, then rejected on compute

MS-SENet (ICASSP 2024, [arXiv:2312.11974](https://arxiv.org/abs/2312.11974),
official Keras code at [MengboLi/MS-SENet](https://github.com/MengboLi/MS-SENet),
temporal backbone from [TIM-Net](https://github.com/Jiaxin-Ye/TIM-Net_SER))
was ported faithfully and trained as a CE baseline on this project's IEMOCAP
protocol. It is **more accurate than DSResNet-SE and not usable as this
project's student**, for reasons that have nothing to do with parameter count.

The port and the feature pipeline are kept (`common/models/mssenet.py`,
`student/precompute_mfcc.py`, `student/train_mssenet.py`) so the result is
reproducible and the comparison can be revisited on faster hardware.

## What was measured

CE only, seed 42, our split (train S2-4 / val S5 / test S1), selection on
validation UA, identical `metrics` code to the DSResNet-SE runs.

| | DSResNet-SE | MS-SENet |
| --- | ---: | ---: |
| Parameters | 96,236 | 148,060 |
| Size, fp32 | 0.37 MiB | 0.57 MiB |
| **MACs / sample** | **35.4 M** | **140.1 M** |
| **Activations / sample** | **1.21 M** | **6.66 M** |
| **Sequential conv layers** | **29** | **50** |
| Seconds / epoch (RTX 3060 Laptop, batch 64) | **1.7** | **320** |
| Best validation UA | 0.5732 | **0.6132** |

The run was stopped at epoch 36 of 70. Its best validation UA of **0.6132 at
epoch 34** was still rising, against **0.574-0.577** for DSResNet-SE under the
same protocol — roughly **+3.6 points**, comfortably outside the +/-1.2 point
run-to-run spread measured on this setup. No test number exists: the selected
checkpoint was in memory when the run was terminated.

**So the accuracy claim stands: at a similar parameter count, this
architecture is genuinely better on our speaker-independent split.**

## Why it was rejected anyway

1.5x the parameters, but **4x the MACs, 5.5x the activations, and 1.7x the
sequential depth** — and 188x the wall-clock time per epoch. The gap between
4x compute and 188x time is the real problem: with 39 channels and 606
timesteps preserved end to end, the work is dominated by kernel-launch
overhead rather than arithmetic (GPU drawing 54 W, 1.8 CPU cores busy, i.e.
launch-bound, not compute-bound).

Three structural causes, all inherent to the design:

- **No temporal downsampling anywhere.** DSResNet-SE shrinks its feature map
  246-fold across its stride-2 blocks (801x64 down to 26x8); MS-SENet holds
  606 timesteps through all 50 conv layers. The official code has its
  `AveragePooling2D` lines commented out — preserving temporal resolution is
  a deliberate design choice of the paper.
- **Long serial chain.** Twenty Temporal-Aware Blocks (ten forward, ten
  backward), two dilated causal convolutions each, all strictly sequential.
  Depth is latency that no amount of GPU width removes.
- **Narrow channels.** `Conv1d(39 -> 39, k=2)` has very low arithmetic
  intensity; fifty such launches leave the GPU waiting on scheduling.

At 320 s/epoch, the planned comparison — CE / logit-KD / feature-KD /
combined-KD across seeds — costs 60+ hours, and the 15 seeds needed to resolve
an effect against this project's measured noise floor would cost several
hundred. That does not fit the remaining schedule.

Untested speedups that would not alter the architecture's mathematics:
`cudnn.deterministic=False` with `benchmark=True` (deterministic dilated
convolution kernels can be several times slower, and this run had determinism
forced on), a larger batch, AMP, and `torch.compile`. Together these might
give 5-10x. They were not pursued because the KD study is the deliverable and
DSResNet-SE already supports it.

## What this is worth to the thesis

A useful negative-space result for the deployment chapter: **parameter count
is a poor proxy for edge cost.** MS-SENet is 0.57 MiB — well inside any
sensible flash budget, and smaller than the 1.8 MB of the supervisor's Coral
system — yet it needs 4x the multiply-accumulates per utterance and refuses to
downsample, which is precisely the property that makes a model expensive on a
microcontroller-class target. A student selected on size alone would have
picked it.

It also gives an honest ceiling: on this split a stronger SER architecture
does reach roughly +3.6 points over DSResNet-SE with cross-entropy alone, so
the DSResNet-SE numbers should be read as a compute-constrained operating
point, not as the best achievable at this scale.

## Reproducing

```bash
python src/iemocap/student/precompute_mfcc.py      # 39-dim MFCC, 22050 Hz, 14.06 s
python src/iemocap/student/train_mssenet.py --seeds 42 --epochs 70
```

Note that `train_mssenet.py` prints only after a seed finishes — an omission
that made this run opaque for three hours. Per-epoch logging should be added
before it is used again.

## Deviations from the common description of the architecture

Verified against the official code, which was followed wherever the two
disagree:

| Frequently stated | Official implementation |
| --- | --- |
| Kernels 9x1, 1x11, 3x3 | **(11,1), (1,9), (3,3)** — Keras feeds [B,T,F,1], so 11 frames along time, 9 bins along frequency |
| Branches concatenated on channels | **Concatenated on axis=2, the frequency axis**; channel count stays 39 throughout |
| SE with a reduction ratio | **Dense(39) -> Dense(39)**, i.e. no reduction |
| Shared frontend, then forward/backward split | **Two independent frontends**, one on the input and one on its time-reversal |
| Residual connections | The TAB gate is **multiplicative**: `F_x = original_x * sigmoid(conv2_out)`, no additive residual |
| — | Frontend output is concatenated with the **raw input**, giving 4F = 156 channels into the temporal stack |

**The official protocol was not reused.** It runs
`KFold(n_splits=10, shuffle=True)` over utterances, which puts the same
speakers in train and test on a ten-speaker corpus, and passes the test fold
in as `validation_data`. Published MS-SENet IEMOCAP figures are therefore not
comparable to anything here.

**The official implementation uses no data augmentation.** Its regularisation
is architectural: SpatialDropout2D 0.2 in the frontend, SpatialDropout1D 0.1
inside each TAB, label smoothing 0.1, and the raw-input skip. (Spatial dropout
zeroes whole channels rather than individual elements, which matters for
convolutional features whose neighbouring positions are highly correlated.)
