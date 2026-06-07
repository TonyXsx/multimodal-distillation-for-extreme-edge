# Multimodal Distillation for Extreme Edge

This repository contains an MSc individual project on distilling task-aware
representations from a prompted multimodal large language model into compact
sensor-only students for extreme-edge deployment.

The current proof of concept focuses on spoken intent classification with
Fluent Speech Commands (FSC). A frozen Qwen2.5-Omni teacher is prompted for the
task, hidden representations are extracted from late LLM layers, a small
teacher-side bottleneck probe is trained, and a tiny audio-only DSResNet-SE
student is trained with label supervision plus knowledge distillation.

## Motivation

Small robots and edge devices often need to react to their surroundings through
cheap sensors and limited compute. Large multimodal models can provide strong
semantic understanding, but they are not deployable on extreme-edge hardware.
This project studies whether a prompted multimodal teacher can transfer useful
task knowledge into a much smaller model that only consumes the sensor modality
available at inference time.

For FSC, the deployed model is audio-only: no text, ASR, or multimodal teacher is
used during inference.

## Method Overview

The pipeline has three stages.

1. Extract teacher representations

   Qwen2.5-Omni is kept frozen. Audio and a task prompt are passed through the
   model, and pooled hidden states are extracted after selected LLM transformer
   blocks. The main selected representation is:

   ```text
   prompt_first_audio_mean_L24-27-30-34
   ```

   In this setting, the prompt appears before the audio tokens, so audio-token
   hidden states can attend to the task instruction under causal masking.

2. Train a compact teacher bottleneck

   A small probe is trained on top of the frozen 2048-dimensional teacher
   representation. The chosen practical bottleneck is:

   ```text
   2048 -> 64 -> 31
   ```

   The 64-dimensional activation is used as the teacher representation target,
   and the classifier output is used as teacher logits.

3. Distill into an audio-only student

   The student consumes 64-bin log-mel spectrograms and outputs both a
   64-dimensional projected representation and task logits:

   ```python
   student_z, student_logits = model(logmel)
   ```

   Training combines cross entropy, logit distillation, and representation
   distillation:

   ```text
   L = CE(student_logits, label)
     + lambda_logit * KL(student_logits / T, teacher_logits / T) * T^2
     + lambda_feature * cosine_distance(student_z, teacher_z)
   ```

## Current Results

### Teacher bottleneck probe

The teacher representation is highly predictive even after strong compression.

| Probe | Architecture | Bottleneck | Eval Acc | Eval Macro F1 |
| --- | --- | ---: | ---: | ---: |
| A1 | 2048 -> 31 | - | 0.9596 | 0.9596 |
| B2 | 2048 -> 64 -> 31 | 64 | 0.9634 | 0.9635 |
| B4 | 2048 -> 256 -> 31 | 256 | 0.9638 | 0.9622 |

The 64-dimensional B2 bottleneck is used as the main representation target for
student distillation.

### Student knowledge distillation

The final held-out FSC test evaluation uses the small DSResNet-SE student:

- Parameters: 97,991
- FP32 size: about 0.37 MB
- Estimated INT8 size: about 0.09 MB
- Input: audio-only log-mel spectrograms

| Method | Test Acc | Test Macro F1 | Test Weighted F1 |
| --- | ---: | ---: | ---: |
| CE-only | 0.9404 | 0.9393 | 0.9404 |
| Logit KD | 0.9684 | 0.9649 | 0.9683 |
| Feature KD | 0.9599 | 0.9585 | 0.9599 |
| Full KD | 0.9631 | 0.9595 | 0.9631 |

These results support the core claim that a prompted multimodal teacher can
improve a tiny, text-free, audio-only student.

## Repository Layout

```text
src/
  common/
    augment.py                 SpecAugment utilities
    config.py                  project-relative paths
    losses.py                  KD losses
    probe.py                   reusable MLP/bottleneck probe
    training.py                evaluation and subset helpers
    models/audio_student.py    DSResNet-SE student model

  fsc/
    frozen_feature_extraction/ Qwen hidden-state extraction
    feature_ablation/          teacher representation selection
    teacher_probe/             bottleneck probe training
    student/                   student KD and final test scripts

  mintrec/                     scaffold for a future multimodal extension

data/                          local datasets, features, checkpoints (gitignored)
outputs/                       result tables and plots (gitignored)
```

## Main Scripts

The FSC pipeline is organized as reusable stages.

```text
src/fsc/frozen_feature_extraction/
  experiment_data_construction.py
  feature_extraction.py
  full_feature_extraction.py

src/fsc/feature_ablation/
  linear_probe.py
  feature_combinations.py

src/fsc/teacher_probe/
  train_probe.py

src/fsc/student/
  precompute_logmel.py
  train_student.py
  tune_kd_hparams.py
  train_student_2x2.py
  final_test.py
```

Generated feature banks, checkpoints, and plots are intentionally excluded from
git because they are large and machine-specific.

## Installation

This project was developed on Windows with CUDA 12.6. Install dependencies with:

```bash
pip install -r requirements.txt
```

The requirements include PyTorch, Hugging Face Transformers/Datasets, audio
processing tools, and plotting/evaluation libraries.

## Reproducing The FSC Experiments

The expected high-level order is:

1. Build or load the FSC data splits.
2. Extract frozen Qwen teacher hidden representations.
3. Run feature ablations to select a teacher representation.
4. Train teacher bottleneck probes.
5. Precompute student log-mel features.
6. Train student KD ablations.
7. Run final held-out test evaluation.

Example entry points:

```bash
python src/fsc/teacher_probe/train_probe.py
python src/fsc/student/train_student_2x2.py
python src/fsc/student/final_test.py
```

The scripts use `src/common/config.py` for project-relative paths, so they should
not depend on hardcoded drive letters.

## Research Direction

FSC serves as the first validation of the idea. The next direction is to extend
the same teacher-representation distillation recipe to richer sensor settings,
such as multimodal intent or interaction datasets, while keeping the deployed
student small and free of text inputs.
