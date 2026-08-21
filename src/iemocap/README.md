# IEMOCAP Track

Third dataset for the thesis, and the one that lines up directly with the
supervisor's own edge work (Kanjo et al., *Transformer Redesign for Late Fusion
of Audio-Text Features on Ultra-Low-Power Edge Hardware*, arXiv:2510.18036):
same corpus, same held-out speakers, same DSResNet-SE student family.

The contrast this track is meant to draw:

| | Kanjo et al. 2025 | This project |
| --- | --- | --- |
| Source of semantic knowledge | frozen KWS branch (LibriSpeech keywords) | frozen omni-MLLM teacher (Qwen2.5-Omni) |
| How it reaches the student | architectural coupling -- branch stays resident at inference | distillation -- knowledge moves into the weights, branch disappears |
| Parameters / size | 734,261 / 1.8 MB | 97,926 / 0.37 MB (FSC student) |
| Knowledge distillation | none | tuned logit + feature KD |

## Data provenance and integrity

The local copy is the full official release (5 sessions + Documentation).
Verified before use:

- All five `Session*.zip` and `Documentation.zip` pass CRC (`unzip -t`).
- 10,039 utterances in `dialog/EmoEvaluation`, matching the official total.
- The 4-class subset comes out at exactly the counts used in the literature
  (angry 1,103 / happy 1,636 / neutral 1,708 / sad 1,084 = 5,531).
- Audio is 16 kHz mono and aligns with the label timestamps (a spot-checked
  clip is 67,982 samples = 4.249 s against a `[5.3661 - 9.6150]` span).

**Video is effectively absent.** Only `Session1/dialog/avi/DivX/` exists, with
28 dialog-level `.avi` files; Sessions 2-5 contain none. Those 28 are ~5-minute
two-speaker conversations that would need cutting by timestamp, they cover only
Session 1 (which is the test split here), and IEMOCAP's on-face motion-capture
markers make them poor input for a vision teacher anyway. **The student on this
track is audio-only** -- which costs nothing in comparability, because the
supervisor's system is audio-only at inference too (its "text" branch is a
keyword spotter running on the same waveform).

## Pipeline

### 1. `data/extract_archives.py`

Selective unpacking. The release inflates to ~20 GB, almost all of it motion
capture, forced alignments, full-dialog wavs and Session 1's videos. Only the
labels, transcripts and utterance-level wavs are extracted: **1.4 GB, 10,341
files**. Idempotent (size-matched files are skipped), so interrupted runs resume.

### 2. `data/build_manifest.py`

Parses labels and transcripts, applies the class filter, assigns the split, and
writes one row per utterance.

**Classes** -- de-facto standard 4-class subset, `exc` merged into `hap`:
`neu->neutral`, `hap`/`exc->happy`, `ang->angry`, `sad->sad`. Dropped: `xxx`
(no annotator majority), `fru`, `sur`, `fea`, `oth`, `dis` -- 4,508 utterances
in total. The consensus label is read from the header line of each
`EmoEvaluation` file; no re-voting.

Deliberately **not** following the supervisor's 5-class setup. His fifth class,
"No Emotion", is synthetic -- MUSAN background noise plus segments where only
the non-target speaker is active -- and is trivially separable, scoring
per-class F1 = 1.0000. It inflates his headline macro-F1 of 0.6107; over the
four real emotion classes it works out to roughly `(5 x 0.6107 - 1.0) / 4 =
0.51`. Sticking to the standard 4 classes keeps this track comparable both to
the wider SER literature and to his *actual* emotion performance.

**Split** -- fixed, speaker-independent, single protocol (no cross-validation,
deliberately, given the remaining project time):

```
train = Sessions 2, 3, 4     val = Session 5     test = Session 1
```

Session 1 is held out for test because that is the set the supervisor reports
on, so accuracy lands on the same speakers. Session 5 is a real validation set
for model selection -- his setup has none, tuning and reporting both on
Session 1. Keeping selection and reporting separate also matches the FSC
chapter's discipline. Every split carries two speakers, so none is decided by a
single voice; the FSC val/test gap (0.82 vs 0.94) showed how far a single
speaker set can move absolute numbers.

Cost of the extra validation session: training drops from 4,446 to 3,205
utterances (-28%). Worth it -- without it the thesis would report on a set it
also selected on.

**Transcripts** -- non-verbal markers are stripped by default. `[LAUGHTER]`,
`[BREATHING]`, `[GARBAGE]`, `[LIPSMACK]` are human annotations rather than
words in the audio, and laughter is close to a giveaway for `happy`: 29 of the
kept utterances have transcripts consisting of *nothing but* a marker, and the
sampled ones are all labelled happy. Left in, the teacher could read the label
off the annotation instead of the speech. 177 markers are removed across the
5,531 kept rows; the 29 now-empty transcripts keep their audio and label. Both
the cleaned and raw text are stored (`transcript`, `transcript_raw`), and
`--keep-markers` regenerates the un-stripped variant for an ablation.

**Scripted vs improvised** -- both kept, matching the supervisor. Every row
carries `is_impro`, so test predictions can be sliced afterwards at zero extra
training cost. This matters because the teacher reads transcripts, and scripted
dialogues reuse fixed lines whose wording correlates with the intended emotion.
If the KD gain is much larger on the scripted slice, the teacher is partly
reciting rather than listening -- the flag is what makes that visible.

## Outputs

```
data/iemocap/extracted/            selectively unpacked release (1.4 GB)
data/iemocap/manifest.csv          5,531 rows x 18 columns
outputs/iemocap/data/dataset_stats.csv        split x class, speakers, hours
outputs/iemocap/data/dataset_stats_impro.csv  split x improvised/scripted
```

Manifest columns: `turn_id, split, session, speaker, dialog, is_impro, emotion,
label, emotion_raw, valence, arousal, dominance, start, end, duration,
wav_path, transcript, transcript_raw`.

### Resulting splits

| split | angry | happy | neutral | sad | total | speakers | hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train (S2-4) | 704 | 916 | 940 | 645 | 3,205 | 6 | 4.02 |
| val (S5) | 170 | 442 | 384 | 245 | 1,241 | 2 | 1.55 |
| test (S1) | 229 | 278 | 384 | 194 | 1,085 | 2 | 1.43 |
| all | 1,103 | 1,636 | 1,708 | 1,084 | 5,531 | 10 | 6.99 |

Speaker-by-split crosstab confirms zero overlap: `Ses01_{F,M}` appear only in
test, `Ses02-04_{F,M}` only in train, `Ses05_{F,M}` only in val.

Two things to carry forward:

- **Class priors differ across splits.** Validation is happy-heavy and
  angry-light (35.6% / 13.7%) while test is neutral-heavy (35.4%). Select on
  unweighted accuracy (macro recall) rather than plain accuracy, and report
  WA + UA + macro-F1 so the numbers stay readable against SER convention.
- **Durations are long-tailed**: mean 4.55 s, p50 3.58, p90 8.69, p95 11.06,
  max 34.14. The fixed-length crop for the student's log-mel needs choosing
  with that in mind (FSC used 3 s, MIntRec 6 s).

## Reproducing

```bash
python src/iemocap/data/extract_archives.py     # ~1.4 GB, resume-safe
python src/iemocap/data/build_manifest.py       # seconds
python src/iemocap/data/build_manifest.py --keep-markers --out-suffix _markers
```

## Roadmap

1. **Teacher feature extraction** -- Qwen2.5-Omni, input ordered
   `instruction -> audio -> transcript -> "Emotion:" readout`. Audio precedes
   the transcript so that under causal masking the audio tokens never see the
   text, which yields a clean audio-only feature and a transcript-aware readout
   from a single forward pass (the arrangement already validated on MIntRec).
   No video.
2. **Frozen-teacher control** -- probe the frozen model's features before any
   adaptation, run at extraction time alongside the main pass. On MIntRec the
   frozen-vs-QLoRA comparison (0.5443 -> 0.5533 on `audio_mean_l27`, against
   0.6130 on `last_token`) is what showed the adaptation's benefit lands almost
   entirely in the text-conditioned readout; the same control is needed here to
   justify the QLoRA step.
3. **QLoRA adaptation**, then extraction of `logits`, `last_token` and
   `audio_mean` for train/val/test.
4. **Feature-KD target**: `last_token` as primary. Teacher-side evidence
   supports it on both previous datasets -- essentially tied on FSC
   (`last_text` 0.9129 vs `audio_mean` 0.9161, ~1 sample apart on a 310-sample
   val set) and clearly ahead on MIntRec (0.6130 vs 0.5533-0.5633). With QLoRA
   the classifier head sits on `last_token`, so logit-KD and feature-KD then
   read from the same point in the network, unlike MIntRec where the two
   targets came from different places. `audio_mean` is kept as a single
   ablation run: it is the direct test of the "unreachable target" worry, and
   IEMOCAP can finally answer it -- MIntRec could not, because every student
   there sat on the macro-F1 ~0.06 noise floor. Note that the readout is
   layer-sensitive (FSC: L34 0.894 -> L27 0.80 -> L24 0.742, while `audio_mean`
   holds 0.90-0.92 across layers), so it must be taken from the final layer.
5. **Student**: DSResNet-SE with the tuned FSC recipe (T=8,
   lambda_logit = lambda_feat = 1.0), plus the augmentation stack the
   supervisor's ablation shows is the single biggest lever there
   (no-aug 0.3366 -> aug 0.4986, before any fusion).
6. **Deployment**: INT8 + latency/memory, against his 1.8 MB / 21-23 ms.

Note when comparing accuracy: his 0.6107 is measured on **re-recorded** Session
1 audio captured through the Coral board's own microphone, on 5 classes
including the synthetic one, with no separate test set. Efficiency numbers
(size, parameters, latency, whether a semantic branch is resident at inference)
compare cleanly; accuracy needs those caveats stated.

## Teacher stage

Three scripts under `teacher/`, all reading utterances through `teacher/data.py`
so the prompt, the input ordering and the label map exist in exactly one place
and cannot drift between the fine-tune and the extraction that reproduces it.

Reused from the MIntRec track rather than reimplemented: `OmniClassifier`,
`load_backbone`, `get_hidden_size`, `LORA_TARGETS` (from
`mintrec/teacher_probe/qlora_finetune.py`), and the shard writer / loader /
resume counter, layer choice and `get_special_id` (from
`mintrec/teacher_probe/extract_features_local.py`). Nothing in those files was
modified. The shard format is therefore unchanged, so existing probe code reads
these features as-is. One side effect: importing the MIntRec extractor pulls in
`cv2`, hence `opencv-python-headless` in the requirements even though this
track has no video.

### `teacher/extract_features.py` -- frozen control AND adapted arm

Both arms share one code path, one prompt, one input order, and the same 4-bit
loader. The only difference is whether LoRA weights are applied, which is what
makes the frozen run a real control rather than a differently-configured run.
(The MIntRec frozen extractor used the opposite input order -- audio last -- so
its frozen-vs-adapted comparison also changed the layout; this one does not.)

Per utterance, one forward pass:

| key | shape | role |
| --- | --- | --- |
| `last_token` | [2048] | readout hidden state, final layer -- primary Feature-KD target |
| `audio_mean_final` | [2048] | clean audio-token mean, final layer |
| `audio_mean_l{24,27,30,34}` | [2048] | clean audio-token mean per layer |
| `audio_mean_L24-27-30-34` | [2048] | mean over those layers (FSC-best combination) |
| `logits` | [4] | head output -- Logit-KD target (adapted arm only) |

Verified locally on the 6 GB RTX 3060, frozen arm, 30 validation utterances:
all seven tensors present, fp16, finite; `audio_mean_l27` and `last_token` have
mean cosine similarity **0.042**, i.e. they really are different
representations rather than near-duplicates. Throughput **~1 s/utterance** on
that card, so the full 5,531 take ~1.5 h locally and roughly 25-35 min on a
4090.

### `teacher/qlora_finetune.py`

4-bit NF4, LoRA r=32 on the seven projection modules, physical batch 1 with
accumulation 16, gradient checkpointing, 3 epochs. Selection is by validation
**UA** (macro recall), not plain accuracy, because the splits have different
class priors. Writes `adapter_epN/` + `head_epN.pt` + `config.json`, and
per-epoch WA/UA/macro-F1 to `outputs/iemocap/teacher_qlora/qlora_<tag>.csv`.
`--no-transcript` trains an audio-only teacher for a modality ablation.

The adapted extraction reads `config.json` back and reproduces the training
input layout, so the two stages cannot disagree about whether the transcript
was present.

### `package_for_upload.py`

Tars only the 5,531 kept wavs plus the manifest: **807 MB uncompressed**
against the 1.4 GB `extracted/` tree. Unpacks straight into a `DATA_ROOT`.

## Running the teacher stage on AutoDL

Image: **PyTorch 2.8.0 / CUDA 12.x / Python 3.10-3.12**, on an **RTX 4090
(24 GB)**. 2.8.0 is the newest offered and closest to the 2.9.1+cu126 used
locally; the 4090 is sm_89 and needs CUDA 12.x. A 3090 works and is cheaper; a
16 GB card would probably fit now that there is no video branch, but the saving
is not worth the OOM risk.

MIntRec needed >=24 GB largely because each sample carried 8 video frames.
IEMOCAP is audio-only at ~4.6 s mean duration, so the fine-tune is far lighter:
expect ~1-1.5 h/epoch over 3,205 utterances, ~3-5 h for three epochs, plus two
extraction passes -- roughly 6-7 GPU-hours in total.

```bash
# on the laptop
python src/iemocap/package_for_upload.py          # -> iemocap_teacher_subset.tar.gz

# on AutoDL, in no-GPU mode first (~0.1 CNY/h): upload, install, fetch weights
bash src/iemocap/setup_autodl.sh
source /root/autodl-tmp/venv/bin/activate
export HF_HOME=/root/autodl-tmp/.hf
tar xzf iemocap_teacher_subset.tar.gz -C /root/autodl-tmp/data

# then start the GPU
python src/iemocap/teacher/extract_features.py --split val --limit 5     # smoke
python src/iemocap/teacher/extract_features.py --split all               # frozen control
python src/iemocap/teacher/qlora_finetune.py --limit 40 --eval-limit 40 --epochs 1
python src/iemocap/teacher/qlora_finetune.py                             # full
python src/iemocap/teacher/extract_features.py --split all \
    --adapter data/iemocap/teacher_qlora/3b_audio-tr_last_r32/adapter_ep3 \
    --head    data/iemocap/teacher_qlora/3b_audio-tr_last_r32/head_ep3.pt
```

Download afterwards: the features are small (5,531 x ~7 x 2048 fp16 is well
under 200 MB per arm) and each adapter is ~330 MB.

Extraction is sharded and resume-safe, so a killed instance costs only the
current shard. The `--limit` smoke runs write into the same shard directory and
simply continue on the full run.

**Model download**: the AutoDL instance sits in mainland China no matter where
you are, so the Qwen2.5-Omni-3B pull happens from there. Nothing is enabled by
default; if it stalls, `source /etc/network_turbo` (AutoDL's own accelerator)
and then `export HF_ENDPOINT=https://hf-mirror.com` are the two fallbacks.
