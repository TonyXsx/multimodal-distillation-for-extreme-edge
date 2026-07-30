# HuBERT-large Audio-only Teacher Baseline (FSC)

## Motivation

The main pipeline (`fsc/frozen_feature_extraction` -> `fsc/teacher_probe` ->
`fsc/student`) shows that distilling from a frozen, prompted Qwen2.5-Omni
teacher improves a tiny audio-only student on FSC (see the main README's
"Current Results"). That comparison alone cannot separate two different
explanations:

1. Any sufficiently strong frozen teacher would improve the student (a
   generic KD effect, unrelated to modality), or
2. The Qwen teacher's multimodal / prompted understanding specifically adds
   value beyond what a strong **audio-only** teacher could provide.

This baseline adds a frozen, audio-only teacher (HuBERT-large) run through
the exact same probe -> KD pipeline as Qwen, so the two teachers can be
compared with the only changed variable being **teacher modality/identity**.
Nothing in `fsc/frozen_feature_extraction`, `fsc/teacher_probe`, or
`fsc/student` is modified -- this baseline lives entirely in this new folder
and reuses the existing student log-mel cache and shared training code by
import.

## Model choice: `facebook/hubert-large-ll60k`

HuBERT-large is trained purely by self-supervised masked prediction of
k-means cluster ids derived from unlabeled audio (Libri-Light, 60k hours) --
**no text or transcripts are involved anywhere in this checkpoint's
pretraining**, so it is a clean audio-only control for the multimodal Qwen
teacher.

Important: this is deliberately the **plain SSL checkpoint**, not
`facebook/hubert-large-ls960-ft` (the CTC-finetuned ASR variant). The `-ft`
checkpoint has been fine-tuned against LibriSpeech transcripts, which would
reintroduce text supervision and defeat the purpose of an audio-only
control.

The SUPERB benchmark's frozen-probe recipe (freeze the SSL encoder, train
only a lightweight downstream head) reports HuBERT-large at 98.76% accuracy
on this exact FSC intent-classification task -- the strongest published
frozen-SSL number for FSC -- which is why it was chosen over wav2vec2-large
as the audio-only control.

## Scope: no ablation, reuse the FSC final configuration as-is

Per the agreed scope for this baseline, no hyperparameter search or
representation ablation is repeated for HuBERT. Every choice below is a
single fixed decision, and every downstream KD constant is imported directly
from the existing `fsc/student` scripts (not retyped), so the two teachers
cannot silently drift apart in anything except teacher identity:

- Feature: mean-pooled over time, **final transformer layer** hidden states
  (no layer/prompt-order sweep -- the Qwen-side ablation already answered
  "which representation", this baseline intentionally skips repeating it).
- Probe: single bottleneck architecture, `hidden_dim -> 64 -> 31`
  (the Qwen pipeline's "B2" config -- the only one the student KD scripts
  actually consume, since Feature-KD's target dim is fixed at 64).
- KD hyperparameters, student architecture, and training recipe: imported
  from `fsc/student/final_test.py` / `fsc/student/kd_common.py` (T=8,
  lam_logit=1.0, lam_feature=1.0, small DSResNet-SE, 70 epochs, AdamW,
  SpecAugment, label smoothing 0.1, seed 42).

## Pipeline

### 1. Feature extraction -- `feature_extraction.py`

Runs frozen HuBERT-large over FSC train + validation (test is intentionally
NOT extracted, mirroring `full_feature_extraction.py`'s rationale: the final
student is audio-only, so test must stay teacher-free). Sample order / ids /
labels are read via `fsc.student.precompute_logmel.load_fsc()` (imported, not
duplicated), so `sample_ids` line up 1:1 with the existing
`data/student/logmel_cache/*.pt` used by the student KD scripts.

```bash
python src/fsc/hubert_baseline/feature_extraction.py
```

Output: `data/teacher_features/fsc_full__hubert-large-ll60k__last_layer_mean/{train,val}_features.pt`

### 2. Teacher probe -- `train_probe.py`

Trains the single `hidden_dim -> 64 -> 31` bottleneck probe with the
**identical** fixed protocol as `fsc/teacher_probe/train_probe.py` (50
epochs, AdamW lr=1e-3/wd=1e-4, batch 256, dropout 0.1, seed 42; standardize
with TRAIN mean/std; report eval accuracy/macro-F1 **once**, on FSC
validation, no early stopping/model selection -- validation is this stage's
final evaluation set, exactly like the Qwen probe).

```bash
python src/fsc/hubert_baseline/train_probe.py
```

Output: probe checkpoint + bottleneck reps under
`data/teacher_probe/fsc_full__hubert-large-ll60k__last_layer_mean/`,
results under `outputs/fsc/hubert_baseline/teacher_probe/`.

### 3. Final KD grid -- `run_final_kd.py`

Trains the same small DSResNet-SE student (97,991 params, ~0.37 MB FP32)
under the 4 methods (CE-only / Logit-KD / Feature-KD / Full-KD), selects the
best-by-val-macro-F1 checkpoint per method, and evaluates each **once** on
the held-out FSC test split -- same protocol as `fsc/student/final_test.py`,
with HuBERT's probe outputs standing in for Qwen's as the KD teacher signal.

```bash
python src/fsc/hubert_baseline/run_final_kd.py
```

Output: `outputs/fsc/hubert_baseline/final_test/results.csv` (+ a
`comparison.csv`/plot overlaying the existing Qwen `final_test/results.csv`
numbers, read-only).

## Results

### Teacher probe (FSC validation, single run, no early stopping)

| Teacher | Architecture | Bottleneck | Eval Acc | Eval Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-Omni (multimodal, prompted) | 2048 -> 64 -> 31 | 64 | 0.9634 | 0.9635 |
| HuBERT-large-ll60k (audio-only, frozen) | 1024 -> 64 -> 31 | 64 | 0.9445 | 0.9467 |

Train accuracy for the HuBERT probe was 0.9974 (vs. eval 0.9445) -- a
noticeably larger train/eval gap than the Qwen probe. This is a direct,
expected consequence of the no-ablation scope: the fixed protocol borrowed
from the Qwen pipeline (50 epochs, single last-layer-mean feature) was not
re-tuned for HuBERT's representation, and is reported as-is rather than
adjusted post hoc.

### Final held-out FSC test (small DSResNet-SE, 97,991 params, ~0.37 MB FP32)

| Method | Qwen Test Acc | Qwen Test Macro F1 | HuBERT Test Acc | HuBERT Test Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| CE-only | 0.9404 | 0.9393 | 0.9438 | 0.9407 |
| Logit KD | 0.9684 | 0.9649 | 0.9647 | 0.9622 |
| Feature KD | 0.9599 | 0.9585 | 0.9568 | 0.9559 |
| Full KD | 0.9631 | 0.9595 | 0.9605 | 0.9582 |

(CE-only is teacher-agnostic; its two columns differ only by ~0.001-0.002,
consistent with ordinary run-to-run/library-version noise despite the same
seed -- it is reported on both sides only as a sanity check, not as a KD
result.)

## Interpretation notes

- **The headline finding**: under this full-data, single-seed, no-ablation
  protocol, the audio-only HuBERT teacher recovers almost all of the KD
  benefit that the multimodal Qwen teacher provides. Relative to the shared
  CE-only baseline, macro-F1 gains are:

  | Method | Qwen gain vs CE | HuBERT gain vs CE | Qwen − HuBERT |
  | --- | ---: | ---: | ---: |
  | Logit KD | +0.0256 | +0.0215 | +0.0041 |
  | Feature KD | +0.0192 | +0.0152 | +0.0040 |
  | Full KD | +0.0202 | +0.0175 | +0.0027 |

  The multimodal/prompted teacher is consistently a little ahead (as the
  main README's numbers would predict), but the gap between the two
  teachers (~0.003-0.004 macro-F1) is small relative to the gain either
  teacher provides over CE-only (~0.015-0.026). In this setting, most of the
  benefit is explained by "distilling from *any* strong frozen teacher", not
  specifically by the teacher being multimodal/prompted.
- This is consistent with, not contradictory to, the project's other
  finding that **KD benefit scales with constraint** (see the 2x2 tuned
  ablation, `fsc/student/kd_2x2_tuned`): FSC full-data with the small
  student is exactly the *low-constraint* corner where KD effects in
  general are smallest and hardest to tell apart. A follow-up under the
  existing 20%-data / small-student constrained settings would be a more
  sensitive test of whether the Qwen-vs-HuBERT gap widens the same way the
  KD-vs-CE gap does -- that would be new ablation work, outside this
  baseline's agreed no-ablation scope, but is the natural next step if the
  modality question needs a sharper answer.
- HuBERT-large's own frozen-probe accuracy under the *matched, no-ablation*
  protocol used here (0.9445) is actually lower than Qwen's B2 probe
  (0.9634), which is the reverse of the literature's frozen-SSL number for
  HuBERT-large on FSC (98.76%, using SUPERB's full learned-weighted-sum
  recipe over all layers, not the single-last-layer mean-pool used here).
  That gap is attributable to the deliberately simpler feature/protocol
  used for this baseline, not to HuBERT being a weaker encoder -- worth
  flagging explicitly so the probe-accuracy numbers above aren't
  misread as "HuBERT is a worse encoder than Qwen for audio", only as
  "under this specific matched pipeline, Qwen's bottleneck was more
  linearly separable than HuBERT's last-layer mean-pool".
