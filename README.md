# Multimodal distillation for extreme edge

This is my MSc individual project. The idea is to use a large multimodal model
as a teacher during training, and deploy only a very small audio-only model that
could run on an extreme-edge device.

The teacher is Qwen2.5-Omni-3B. It is given the audio, and depending on the
dataset the transcript and video as well, together with a prompt describing the
task. Instead of using the text it generates, I take its hidden states, pool
them, and compress them into a 64-dimensional target. A student of under 100k
parameters is then trained against that target from a log-mel spectrogram alone.
At inference the student sees no text, no video and no teacher. The transcript
is privileged information: it is there while the student trains and gone when it
runs.

Three datasets were used, in this order. Fluent Speech Commands is the proof of
concept, MIntRec 2.0 is a harder task that the student could not handle, and
IEMOCAP is the main study.

## Method

The pipeline is the same on all three datasets and has three stages.

The teacher is either frozen or adapted with LoRA. Audio and a task prompt go
in, and the hidden states after a few late transformer blocks come out, pooled
over the audio tokens. Two readouts are used: the audio-token mean, which cannot
see the transcript, and the last-token readout, which can.

A small probe is then trained on the frozen 2048-dimensional feature, with a
bottleneck of 64 in the middle. The bottleneck activation is the feature target
and the probe output is the logit target. Both are fitted on the training split
only, so the student never gets a teacher signal on validation or test.

Finally the student is trained. It takes a 64-bin log-mel spectrogram and
returns a 64-dimensional embedding and class logits, and its loss is
cross-entropy plus a KL term against the teacher logits plus a cosine term
against the teacher embedding. The student is a DSResNet-SE with about 97k
parameters, roughly 0.37 MB in fp32.

## Fluent Speech Commands

FSC has 31 intent classes and is easy enough to check the whole pipeline end to
end, so the teacher stays frozen here. The feature was chosen by a probe
ablation over layer, pooling and prompt order; the best one is the mean of
layers 24, 27, 30 and 34 over the audio tokens with the prompt placed first.
Compressing it to 64 dimensions costs almost nothing, and the probe reaches
0.9634.

On the held-out test split the student goes from 0.9404 with cross-entropy to
0.9684 with logit distillation, with feature and full distillation in between.
A 2x2 ablation over student size and training data shows the gain growing as the
student gets more constrained: about 1 point with the strong student and all the
data, about 6 points with the small student and a fifth of it.

A HuBERT-large control teacher was run through the same pipeline to see whether
the gain really needs a multimodal teacher. It reaches 0.9647 against Qwen's
0.9684, which is too close to call on a single run, so FSC cannot separate the
two teachers.

## MIntRec 2.0

MIntRec 2.0 is 30-class conversational intent from TV dialogue, with audio,
video and transcript. A frozen probe only reaches 52 to 58 percent here, so the
teacher was adapted with QLoRA instead. It reaches 0.653 test accuracy, above
the fusion baselines published with the dataset (0.593 to 0.607) and above a
recent method for the task at 0.642, which makes adapting a general model a
cheap way to get a strong teacher.

The student side did not work. The audio-only student goes from 0.1087 with
cross-entropy to 0.1299 with full distillation, against a chance level of 3.3
percent, and an audio-visual variant is worse than the audio-only one on every
matched comparison. The predictions are spread over most classes rather than
collapsed, so this looks like a capacity limit: the task is largely text-driven
and 100k parameters trained from scratch cannot recover it from audio. I kept
this in the repo as a negative result, and it is why IEMOCAP became the main
track.

## IEMOCAP

IEMOCAP is four-class emotion recognition over 5,531 utterances. It sits between
the other two in difficulty and prosody carries much of the label, so an
audio-only student is not at an obvious disadvantage.

Most of the early work here went into the protocol. Rerunning the same
configuration changed the result by more than the effect being measured, and
picking the best epoch on validation favoured the cross-entropy baseline by
about a point. So checkpoint selection was dropped, the epoch count fixed, and
every method run with five seeds against a frozen baseline, evaluated with
five-fold leave-one-session-out. The runs from before that are archived rather
than deleted, since they are the evidence that the measurement was the problem.

Under that protocol, feature distillation against the teacher bottleneck does
essentially nothing on its own. Looking at how closely the student reproduces
the target explains why: trained jointly with cross-entropy it gets 73 percent
of the way there on the training split and 10 percent on test, while the cosine
term alone gets 69 percent and 21 percent. Cross-entropy gives a small encoder a
cheaper way to lower the loss, which is to memorise the training set instead of
approximating the teacher.

That leads to a two-stage version. The encoder is trained against the teacher
bottleneck with the cosine term alone and no labels at all, then frozen, and a
64 to 64 to 4 head of 4,420 parameters is fitted on the cached embeddings.
Labels only enter through that head.

| teacher and target | UA | vs CE | p |
| --- | --- | --- | --- |
| CE baseline | 0.5245 | | |
| Qwen, probe target | 0.5483 | +2.38 | 0.0002 |
| HuBERT, probe target | 0.5773 | +5.27 | 0.0015 |
| HuBERT, PCA target | 0.5894 | +6.49 | 0.0005 |

Two results came out of this that I did not expect. The first is that
HuBERT-large, which is audio-only and about fourteen times smaller than Qwen,
transfers better on every arm, even though Qwen is the better teacher at the
task itself. The second is that replacing the label-trained probe with a plain
PCA projection of the same hidden state, using no labels at all, improves things
further. The best configuration in the study is the audio-only teacher with the
unsupervised target, 6.49 points above the baseline, at a teacher-to-student
parameter ratio of roughly 49,000 to 1.

## Repository

```
src/
  common/    losses, probe, training loop and the student model
  analysis/  how much information the teacher soft labels carry
  fsc/       the FSC track
  mintrec/   the MIntRec 2.0 track
  iemocap/   the IEMOCAP track, including the diagnostics above
  early_experiment/  the first notebook, kept for reference

data/        datasets, features and checkpoints (gitignored)
outputs/     result csvs, figures and notes (gitignored)
```

Each track has the same three stages as the method above, plus an analysis stage
for IEMOCAP, and the folders under it are named after them. The tracks were
written months apart and some code is duplicated between them on purpose, so
that a change made for one dataset cannot quietly move the numbers of another.


## Setup

Local development was on Windows with an RTX 3060, using `requirements.txt`.
The teacher stages do not fit on that GPU, so they were run on rented Linux
GPUs: MIntRec on RunPod and IEMOCAP on AutoDL, each with its own requirements
file and setup script. Both scripts build a virtual environment on the
persistent volume and inherit the image's torch instead of reinstalling it.
Paths are all relative, and `data/` has to be created and downloaded by the user because of privacy issue.


