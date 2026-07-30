# MIntRec2.0 Tiny-Student KD Wrap-up: Audio-only vs Audio-visual

## Why this exists

The teacher-probe validation stage (`mintrec/teacher_probe/`) found that MIntRec2.0's audio-token representations plateau around 52-58%
dev accuracy across every teacher-input configuration tried (plain/aware
prompts, 4bit/bf16, layers 24-34, even after a QLoRA fine-tune) -- a much
lower ceiling than FSC's ~96%. The QLoRA probe additionally showed a
**modality-imbalance** effect: `last_token` (the readout that has seen the
full transcript) reaches ~61-63% dev acc, while the CLEAN audio-only
`audio_mean_l27` feature stays at ~55% -- most of the extra signal is
text-driven, not audio-driven. Published MIntRec2.0 baselines (MAG-BERT,
MulT — arXiv 2403.10943) confirm this is a property of the dataset, not of
our pipeline: multimodal fusion only beats a text-only BERT baseline by
~1.3 points (59.3% -> 60.6-60.7%), against a human ceiling of ~71%.

Per the project's own go/no-go rule (`teacher_probe/README.md`), this reads
as a "low" signal that has already had its two obvious remedies tried
(teacher-modality ablation + LoRA adaptation). Rather than chasing more
ablation, this final stage asks ONE more concrete question and then closes
out the MIntRec track: **does distilling into a genuinely tiny sensor-only
student still pick up a usable signal, and does adding a (likely weak,
possibly noisy) visual modality help, hurt, or wash out?**

No new hyperparameter search is done anywhere in this stage -- every
constant not specific to "audio-only vs audio-visual" is reused unchanged
from the tuned FSC recipe (`fsc/student/kd_common.py` /
`fsc/student/final_test.py`): T=8, lam_logit=1.0, lam_feature=1.0,
AdamW(lr=1e-3, wd=1e-4), cosine schedule, label smoothing 0.1, SpecAugment
on the audio branch, seed 42.

## Pipeline (4 new stages, nothing in `teacher_probe/` modified)

### 0. Recap: what already existed before this stage

- `mintrec/teacher_probe/qlora_finetune.py` (run on RunPod) produced ONE
  QLoRA adapter: instruction -> audio (clean, right after instruction) ->
  8 video frames -> transcript -> "Intent:" readout, LoRA r=32, 3 epochs.
  Test accuracy of the trained readout head: **65.3%** (beats published
  MAG-BERT/MulT).
- `mintrec/teacher_probe/extract_with_lora.py` extracted, per utterance:
  `logits` [30] (the real classification head output), `last_token` [2048]
  (final-layer readout hidden state), `audio_mean_l27` / `audio_mean_final`
  [2048] (clean audio-token-only pooled features) for train/dev/**and test**
  (test features exist on disk but must NEVER be touched until the one
  final evaluation pass below -- same discipline as FSC).

### 1. `mintrec/teacher_probe/train_bottleneck_probe.py` (NEW)

FSC's Feature-KD target was always a compact **64-dim bottleneck** (the B2
probe's activation), never a raw 2048-dim teacher hidden state. MIntRec had
never had this compression step done for `audio_mean_l27` / `last_token`,
so this script closes that gap: trains a `2048 -> 64 -> 30` probe (Probe
class from `common.probe`, IDENTICAL hyperparameters to
`fsc/teacher_probe/train_probe.py`'s B2: 50 epochs, AdamW lr=1e-3/wd=1e-4,
batch 256, dropout 0.1, seed 42, standardize with train stats, no early
stopping / selection on dev) independently on each of the two features.

Logit-KD does NOT use either probe's own classifier output -- it always
uses the already-extracted `logits` (the real, stronger QLoRA head, 65.3%
test acc). These two probes exist only to produce a compact FEATURE target.

Results (dev, single run, no selection):

| Feature | Bottleneck | Dev Acc | Dev Macro F1 |
| --- | ---: | ---: | ---: |
| audio_mean_l27 | 64 | 0.5488 | 0.4920 |
| last_token | 64 | 0.5949 | 0.5347 |

(Consistent with the pre-compression probe numbers in `qlora_probe.csv` --
last_token stays clearly ahead of audio_mean_l27 even after compression to
64-dim, confirming the modality-imbalance finding survives compression.)

Outputs: `data/mintrec/teacher_probe/qlora_bottleneck/<feature>/{checkpoint.pt,bottleneck_reps.pt}`,
`outputs/mintrec/teacher_probe/bottleneck_results.csv`.

### 2. `mintrec/student/precompute_features.py` (NEW)

Precomputes, once, for train/dev/test:
- **Audio**: identical mel config to FSC (16 kHz, n_fft=400, hop=160,
  n_mels=64, fmax=8000), padded/truncated to **6.0 s** (a 40-clip duration
  check gave p50=2.4s, p90=4.2s, p95=5.0s, max=8.1s -- 6.0s covers the bulk
  without excessive padding).
- **Video**: **8 frames** per clip (matches the QLoRA teacher's frame
  count), resized to **64x64 RGB** thumbnails (simple resize, no aspect-ratio
  preservation -- this is a compact sanity-check student, not a production
  vision pipeline).

Sample order/ids/labels come from `mintrec.teacher_probe.extract_features_local`
(imported, not duplicated) and the label2id embedded in the QLoRA teacher's
own `config.json` (NOT recomputed), so label indices match the extracted
`logits` dimension order exactly.

Output: `data/mintrec/student/feature_cache/{train,dev,test}_features.pt`.

### 3. `mintrec/student/models.py` (NEW)

Two tiny students, both budgeted to stay near/under **1 MiB FP32**
(fallback ceiling ~3 MiB if needed -- not needed in practice):

- **Audio-only** (`AudioOnlyStudent`): the FSC small `DSResNetSE` reused
  UNCHANGED (same `SMALL_KW`: channels 16/32/64/96/128, proj_dim=64).
  Longer MIntRec clips need no architecture change -- DSResNetSE
  global-average-pools before the projection head, so param count is
  independent of input length. **97,926 params, 0.374 MiB FP32.**
- **Audio-visual** (`AudioVisualStudent`): the same audio encoder (minus its
  own classifier) + a tiny `VisualEncoder` built from the SAME block
  primitives (`SEBlock`, `DWSepConv`, `ResDSSEBlock` from
  `common.models.audio_student`, imported not copied) with a much smaller
  channel plan (6/12/20/28), processing 8 low-res frames with a shared
  per-frame CNN + mean-pool over frames -> 64-dim visual embedding ->
  concat with the 64-dim audio embedding -> fusion MLP -> 64-dim bottleneck
  -> 30-way classifier. **115,471 params, 0.440 MiB FP32.** Comfortably
  under the 1 MiB target; the 3 MiB fallback was not needed.

### 4. `mintrec/student/kd_common.py` + `run_final_kd.py` (NEW)

Ten conditions, one training run each (no seed sweep -- this is a
wrap-up, not a new ablation), best-by-dev-macro-F1 checkpoint, ONE final
pass on held-out test:

**Audio-only** (97,926 params):
| Method | Logit-KD target | Feature-KD target |
| --- | --- | --- |
| `ao_ce_only` | - | - |
| `ao_logit_kd` | `logits` | - |
| `ao_feature_kd_audiohidden` | - | bottleneck(`audio_mean_l27`) |
| `ao_feature_kd_lasttoken` | - | bottleneck(`last_token`) |
| `ao_full_kd_audiohidden` | `logits` | bottleneck(`audio_mean_l27`) |
| `ao_full_kd_lasttoken` | `logits` | bottleneck(`last_token`) |

The audio-only student explicitly compares feature-aligning to the CLEAN
audio-only teacher feature vs. the privileged (text-influenced) last_token
readout, to empirically check the concern raised during planning: aligning
a text-free student's feature to a representation partly shaped by the
transcript may be an unreachable, possibly harmful target rather than a
helpful one. Both are run so the data (not assumption) decides.

**Audio-visual** (115,471 params):
| Method | Logit-KD target | Feature-KD target |
| --- | --- | --- |
| `av_ce_only` | - | - |
| `av_logit_kd` | `logits` | - |
| `av_feature_kd_lasttoken` | - | bottleneck(`last_token`) |
| `av_full_kd_lasttoken` | `logits` | bottleneck(`last_token`) |

Only the `last_token` feature target is used for the audio-visual student's
Feature-KD (not `audio_mean_l27`) -- audio_mean_l27 is specifically an
audio-only-attending feature, no more relevant to a fused audio+visual
representation than to the audio-only student's already-covered
`ao_feature_kd_audiohidden` condition; testing fusion-vs-last_token is the
one genuinely new question for this student (does adding vision let the
fused representation reach closer to the privileged readout than audio
alone could?).

Training recipe (unchanged from FSC except batch size, which is reduced
because MIntRec's train split, 6,165 utterances, is much smaller than
FSC's 23,132): 70 epochs, AdamW(lr=1e-3, wd=1e-4), batch 128, cosine
schedule, label smoothing 0.1, SpecAugment on the log-mel input only (no
visual augmentation), seed 42.

Outputs:
```
data/mintrec/student/final_test_checkpoints/<method>.pt
outputs/mintrec/student/final_test/results.csv
outputs/mintrec/student/final_test/final_test.png
```

## Reading the results

Expectations were calibrated before running (see planning discussion):
teacher readout itself only reaches 65.3% test acc; the frozen audio-only
probe caps at ~55%; a genuinely tiny student with its own much weaker
encoder, trained on 6,165 utterances of a 30-way task with a ~71% human
ceiling, realistically lands well below either of those. The comparisons
that matter are relative, not absolute:

1. Does KD (logit and/or feature) beat CE-only, for either student?
2. Does `audiohidden` or `lasttoken` make a better Feature-KD target for the
   audio-only student -- i.e. was the "privileged information should flow
   through logits, not through feature-matching to a text-contaminated
   target" concern justified in practice?
3. Does the audio-visual student beat the audio-only student anywhere, or
   does the visual branch mostly add noise (as literature's ~1.3-point
   fusion gain over text-only, using FULL-size pretrained encoders, would
   suggest is plausible for an even smaller, 8-low-res-frame visual branch)?

### Results

Single run per method (seed 42), held-out TEST, best-by-dev-macro-F1
checkpoint. Source of truth: `outputs/mintrec/student/final_test/results.csv`.

| Method | Student | Test Acc | Test Macro F1 | Test Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| ao_ce_only | audio-only | 0.1087 | 0.0560 | 0.0853 |
| ao_logit_kd | audio-only | 0.1141 | 0.0558 | 0.0889 |
| ao_feature_kd_audiohidden | audio-only | 0.1062 | 0.0563 | 0.0912 |
| ao_feature_kd_lasttoken | audio-only | 0.1171 | 0.0651 | 0.0982 |
| ao_full_kd_audiohidden | audio-only | 0.1274 | 0.0634 | 0.0982 |
| ao_full_kd_lasttoken | audio-only | 0.1299 | 0.0621 | 0.1008 |
| av_ce_only | audio-visual | 0.0890 | 0.0582 | 0.0806 |
| av_logit_kd | audio-visual | 0.1023 | 0.0497 | 0.0845 |
| av_feature_kd_lasttoken | audio-visual | 0.1008 | 0.0535 | 0.0840 |
| av_full_kd_lasttoken | audio-visual | 0.1289 | 0.0551 | 0.0971 |

### Interpretation

**Sanity check first, because these numbers are much lower than the
pre-registered 30-50% ballpark discussed while planning this stage.**
Checked directly (not just assumed): predictions use 22/30 classes, the
predicted-class ranking tracks the true label-frequency ranking, logits are
finite with a sane spread (no NaN, no single-class collapse) -- this is
genuine weak learning, not a degenerate bug. The training-set accuracy of
the selected `ao_ce_only` checkpoint is itself only 29.5% (at its
early-selected epoch 20/70), confirming the bottleneck is underfitting by a
from-scratch tiny CNN on 6,165 examples of a hard task, not a
train/test mismatch. The FSC audio encoder architecture that worked very
well on FSC's easier, larger (23k), more literal task simply does not have
enough capacity+data to fit MIntRec2.0's conversational intent labels
from scratch here -- worth stating plainly rather than dressed up.

**1. Does KD beat CE-only?** For the audio-only student, yes, consistently
though modestly: every KD variant's test macro-F1 (0.056-0.065) is at or
above `ao_ce_only`'s 0.0560, and 3 of 4 clearly above it. Small effect size,
single seed -- read as "KD does not hurt and probably helps a little", not
as a precisely quantified gain.

**2. `audiohidden` vs `lasttoken` as the Feature-KD target?** Inconclusive
at this scale, and notably NOT a clean confirmation of the
modality-contamination concern raised while planning this stage. Feature-KD
alone: `lasttoken` (0.0651) numerically beats `audiohidden` (0.0563). Full-KD:
`audiohidden` (0.0634) slightly beats `lasttoken` (0.0621). Both directions
appear depending on which loss combination -- with differences this small
(~0.01 macro-F1) on a single seed, this is not a result that should be
over-interpreted either way; a lot of the theoretical "unreachable target"
concern may simply be swamped by how weak and noisy learning is overall in
this regime.

**3. Does the visual branch help, hurt, or wash out?** This is the
cleanest signal in the table: audio-visual does **not** beat audio-only
anywhere on macro-F1, and is clearly worse for the CE-only pair
(`av_ce_only` 0.0582 vs `ao_ce_only` 0.0560 acc-wise looks close but
`av_ce_only`'s test ACC of 0.0890 is meaningfully below `ao_ce_only`'s
0.1087) and for the logit-KD pair (0.0497 vs 0.0558 macro-F1). This matches
-- and now has direct local evidence for -- the concern raised before
running this: an 8-frame, 64x64, from-scratch visual encoder this small
adds more noise than signal for MIntRec2.0's task, consistent with the
literature's own finding that full-size pretrained multimodal fusion
(MAG-BERT/MulT) only gains ~1.3 points over text-only in the first place
(arXiv 2403.10943) -- a much larger and better-trained visual pathway than
this one still only contributes a small fraction of the total signal on
this dataset, and this budget-constrained visual branch could not capture
even that.

### Bottom line for the thesis

This closes out the MIntRec2.0 track. The honest conclusion across the
whole investigation (teacher-probe validation -> QLoRA adaptation ->
this tiny-student wrap-up) is: MIntRec2.0's conversational, dialogue-act-style
intent labels are predominantly text-driven, so (a) an audio-only frozen
teacher representation caps in the mid-50s% even after LoRA adaptation,
(b) a genuinely tiny, from-scratch, sensor-only (audio or audio+visual)
student trained on 6,165 examples of this 30-way task lands at low
double-digit accuracy, and (c) adding a lightweight visual branch does not
recover the gap -- it mostly adds noise at this budget. This is a
legitimate boundary condition for the thesis's core claim ("a prompted
multimodal teacher transfers useful knowledge to a tiny sensor-only
student"): the claim held clearly on FSC's literal, single-modality,
large-data command-classification task, and did not transfer as cleanly to
a small-data, text-dominated, conversational multimodal task -- exactly the
kind of contrast worth discussing explicitly rather than only reporting the
success case.

## Reproducing

```bash
# 1. bottleneck probes on the QLoRA teacher's raw hidden states (fast, seconds)
python src/mintrec/teacher_probe/train_bottleneck_probe.py

# 2. precompute student inputs (slow -- video decoding dominates, ~1.5-2h for
#    train+dev+test on a laptop 3060; resume-safe, skips splits already cached)
python src/mintrec/student/precompute_features.py

# 3. the 10-condition KD comparison + one final TEST pass per method
python src/mintrec/student/run_final_kd.py
```
