# Multimodal distillation for extreme edge

This is my MSc individual project. The question is whether a large multimodal
model can be used as a teacher to train a very small, sensor-only model that
could run on extreme-edge hardware.

The setup is the same in every experiment. A large model, Qwen2.5-Omni-3B, is
given the audio (and for one dataset the video and transcript as well) together
with a prompt describing the task. Its hidden states are pulled out and turned
into a distillation target. A student of about 100k parameters is then trained
against that target using only a log-mel spectrogram. At inference the student
sees nothing but audio: no text, no video, no teacher.

Three datasets were used, in this order. Fluent Speech Commands was the proof of
concept, MIntRec 2.0 was a harder multimodal task that the student could not
handle, and IEMOCAP is the main study.

## Method

There are three stages.

First, extract teacher features. The teacher is frozen for FSC and LoRA
fine-tuned for MIntRec and IEMOCAP. Audio and a task prompt go in, and the
hidden states after a few selected transformer blocks come out, pooled over the
audio tokens. The prompt is put before the audio, so under causal masking the
audio tokens can attend to the instruction.

Second, train a small probe on the frozen 2048-dimensional teacher feature, with
a bottleneck in the middle:

```
2048 -> 64 -> n_classes
```

The 64-dimensional activation is used as the feature target and the probe output
is used as the logit target. Both are fitted on the training split only, so the
student never receives a teacher signal on val or test.

Third, train the student. It takes a 64-bin log-mel spectrogram and returns a
64-dimensional embedding z and class logits:

```
loss = CE(logits, y)
     + lam_logit   * KL(logits / T, teacher_logits / T) * T^2
     + lam_feature * (1 - cos(z, teacher_z))
```

The student is a DSResNet-SE with around 97k parameters, about 0.37 MB in fp32
and roughly 0.09 MB if quantised to int8.

## Fluent Speech Commands

FSC has 31 intent classes and is easy enough that the whole pipeline can be
checked end to end. The teacher stays frozen here.

The teacher feature was picked by a linear probe ablation over layer, pooling
and prompt order. The winner was the mean of layers 24, 27, 30 and 34, pooled
over the audio tokens, with the prompt placed first (0.9226 val accuracy). A
2048->64->31 probe on that feature reaches 0.9634, so almost nothing is lost by
compressing down to 64 dimensions.

KD hyperparameters were tuned on val and ended up at T = 8, lam_logit = 1.0 and
lam_feature = 1.0. On the held-out test split:

| method | test acc | test macro F1 |
| --- | --- | --- |
| CE only | 0.9404 | 0.9393 |
| logit KD | 0.9684 | 0.9649 |
| feature KD | 0.9599 | 0.9585 |
| full KD | 0.9631 | 0.9595 |

A 2x2 ablation over student capacity and training data size, 3 seeds each, shows
the benefit growing as the student gets more constrained. With the strong student
and all the data, logit KD adds about 1.2pp over CE. With the small student and
20% of the data it adds about 6.3pp. The data limit matters more than the
capacity limit.

There is also a HuBERT-large control teacher for FSC, to check how much of the
gain really needs a multimodal model. It gets 0.9647 with logit KD against
Qwen's 0.9684, which is close enough that FSC cannot separate the two.

## MIntRec 2.0

MIntRec 2.0 has 30 intent classes and real multimodal input, so it was meant to
be the harder version of the same idea. The teacher was QLoRA fine-tuned and
reaches 0.6275 dev accuracy, and its last-token feature probes to 0.5949 through
a 64-dimensional bottleneck, so the teacher side works.

The student side does not. The audio-only student goes from 0.1087 test accuracy
with CE to 0.1299 with full KD, and the audio-visual student from 0.0890 to
0.1289. Distillation does help in relative terms, but a 100k-parameter model has
nowhere near the capacity for this task and the absolute numbers are too low to
draw a conclusion from. This is kept in the repo as a negative result rather than
deleted, and it is the reason the project moved on to IEMOCAP.

## IEMOCAP

IEMOCAP is the main study. It has 4 emotion classes, 5,531 utterances and about
7 hours of audio from 10 speakers, which is small enough that measurement noise
becomes the main problem.

Most of the work here went into fixing the protocol before comparing anything.
Selecting the checkpoint on val turned out to flip results by up to 6.8pp, and it
favoured CE over KD by about 1pp, so checkpoint selection was dropped entirely
and every student now runs for a fixed 70 epochs. Each method runs 5 seeds,
paired against the CE baseline with the same config, and results append to one
csv with a config hash so CE never has to be re-run. Earlier runs that used the
old protocol are archived under `outputs/iemocap/student/exploratory/` instead of
being deleted, because they are the evidence that the measurement was the
problem, not the method.

Three splits are supported through the `IEMOCAP_PROTOCOL` environment variable:
speaker-independent, speaker-dependent, and leave-one-session-out. The headline
numbers use 5-fold LOSO with 5 seeds per fold, paired across folds against a
shared CE baseline of 0.5245 test UA:

| teacher | method | test UA | delta vs CE | p |
| --- | --- | --- | --- | --- |
| HuBERT | two-stage | 0.5773 | +5.27pp | 0.0015 |
| HuBERT | logit KD | 0.5719 | +4.74pp | 0.0012 |
| Qwen | two-stage | 0.5483 | +2.38pp | 0.0002 |
| Qwen | full KD | 0.5348 | +1.03pp | 0.087 |
| Qwen | logit KD | 0.5318 | +0.73pp | 0.406 |
| Qwen | feature KD | 0.5250 | +0.05pp | 0.937 |

Two things came out of this.

The first is the two-stage method. Instead of training the student on CE and KD
together, the encoder is trained with the cosine term alone and no labels at all,
then frozen, and a small 64->64->4 head is fitted on the resulting embeddings
with CE plus logit KD. That head has 4,420 parameters, so labels only enter the
model through it. This is the best method under both teachers and it has no KD
hyperparameter left to tune.

The second was not the plan. A HuBERT-large audio-only teacher was added as a
control, expecting it to show that the multimodal teacher was doing the work. It
distils better than Qwen instead, and by a clear margin, even though the Qwen
target wins every direct measurement of target quality. Most of
`outputs/iemocap/analysis/` is the follow-up to that, looking at what the
64-dimensional target actually contains and how much of it the student can
reproduce.

## Layout

```
src/
  common/          shared losses, probe, training loop, student model
  analysis/        how much information the teacher soft labels carry
  fsc/             the FSC pipeline, plus the HuBERT control and a gradio demo
  mintrec/         the MIntRec 2.0 pipeline
  iemocap/         the IEMOCAP pipeline: teacher, student, protocol, analysis

data/              datasets, features, checkpoints (gitignored)
outputs/           result csvs, figures and notes (gitignored)
```

Each dataset has its own teacher stage, its own student stage, and IEMOCAP also
has an analysis stage. Some code is duplicated between them on purpose: the three
pipelines were written months apart, and I did not want a change made for one
dataset to quietly move the numbers of another.

## Setup

Local development was on Windows with an RTX 3060, CUDA 12.6, using
`requirements.txt`. The teacher stages do not fit on that GPU, so they were run
on rented Linux GPUs. MIntRec used RunPod with `requirements-runpod.txt` and
`setup_runpod.sh`; IEMOCAP used AutoDL with `src/iemocap/requirements-autodl.txt`
and `src/iemocap/setup_autodl.sh`. Both setup scripts build a venv on the
persistent volume and inherit the image's torch instead of reinstalling it.

Paths are relative and come from `src/common/config.py` and
`src/iemocap/paths.py`, so nothing depends on a drive letter. `data/` is a
symlink here because the datasets do not fit on the system disk.

## Running things

The FSC pipeline runs in this order:

```
python src/fsc/frozen_feature_extraction/experiment_data_construction.py
python src/fsc/frozen_feature_extraction/full_feature_extraction.py
python src/fsc/feature_ablation/linear_probe.py
python src/fsc/feature_ablation/feature_combinations.py
python src/fsc/teacher_probe/train_probe.py
python src/fsc/student/precompute_logmel.py
python src/fsc/student/tune_kd_hparams.py
python src/fsc/student/train_student_2x2.py
python src/fsc/student/final_test.py
```

IEMOCAP is the same shape, except the protocol and the teacher are chosen with
environment variables:

```
python src/iemocap/data/build_manifest.py
python src/iemocap/teacher/lora_finetune.py
python src/iemocap/teacher/extract_features.py --adapter ...
python src/iemocap/teacher_probe/probe_features.py
python src/iemocap/student/precompute_logmel.py

IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/run_fixed_protocol.py --methods ce
IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/run_fixed_protocol.py --methods feature_only_audio
IEMOCAP_PROTOCOL=loso1 python src/iemocap/student/stage2_readout.py
```

Setting `IEMOCAP_TEACHER=hubert` switches to the control teacher. Every artefact
gets a suffix for the protocol and the teacher, so runs never overwrite each
other.

## Results

The scripts write their numbers straight to csv under `outputs/`. The files worth
reading first are `outputs/iemocap/student/RESULTS.md`,
`outputs/iemocap/analysis/teacher_compare/FINDINGS.md` and
`outputs/fsc/student/final_test/results.csv`.

## Limitations

The IEMOCAP teacher has to be LoRA fine-tuned separately for every fold, which is
why the earlier work used a single split rather than full cross validation. The
LOSO numbers fix that, but the stage-2 head was chosen after the first results
were already seen, so the candidate set was not pre-registered even though the
selection itself was made on val. MIntRec 2.0 is unfinished in the sense that no
student of this size gets near a usable accuracy. Nothing has been deployed on
real edge hardware yet, so the size figures are estimates from parameter counts
rather than measurements.
