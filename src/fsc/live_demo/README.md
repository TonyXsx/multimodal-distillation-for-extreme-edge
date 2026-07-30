# Live Microphone Demo (Gradio)

A small end-to-end sanity check: speak into your own microphone, run the
audio straight through the trained tiny FSC student, see which of the 31
intents it predicts.

**This is a qualitative demo, not a formal evaluation.** Do not use its
output to update any results table -- the FSC test-set numbers in the main
README and in `fsc/hubert_baseline/README.md` remain the source of truth.
This tool exists to (a) sanity-check the end-to-end pipeline by ear, and
(b) get a feel for practical robustness outside the FSC recording
conditions.

## Why live-mic accuracy will likely be lower than the reported test accuracy

FSC was recorded by 97 fixed speakers under fairly consistent microphone/room
conditions. Your voice, microphone, room acoustics, and phrasing are outside
that training distribution, so accuracy here can be noticeably lower than
the ~94-97% reported on the held-out FSC test split. That is an expected
domain-shift effect, not evidence that the KD approach itself is broken --
if anything, it is a legitimate limitation to note in the thesis
(this student was never trained for speaker/environment generalization).

## The model only understands 31 fixed smart-home intents

FSC is a closed-vocabulary dataset. The classifier always outputs a
probability distribution over the same 31 classes, no matter what you say
-- there is no "none of the above". For the test to be meaningful, speak
something close to one of the actual FSC sentences, e.g.:

```
Turn on the kitchen lights          Turn off the lamp
Turn the lights on                  Resume
Turn off the music                  Turn the volume down
Volume up                           Turn up the temperature in the bedroom
Turn the kitchen temperature down   Bring me my shoes
Get me the newspaper                Set language to Chinese
```

Full list of 31 intents: `data/fsc_small_ablation/config.json` ->
`intent_labels`.

## What it reuses (nothing here re-derives the training pipeline)

- Mel extraction: `fsc.student.precompute_logmel.wav_to_logmel` (16 kHz,
  n_fft=400, hop=160, n_mels=64, fmax=8000, pad/truncate to 3 s) -- imported
  directly, not reimplemented, so the live pipeline cannot silently diverge
  from what the student was trained/evaluated on.
- Normalization: the exact train-set per-bin mean/std saved in
  `data/student/logmel_cache/train_logmel.pt`.
- Model: `common.models.audio_student.DSResNetSE` with
  `fsc.student.final_test.SMALL_KW` -- the same small student
  (97,991 params, ~0.37 MB FP32) used for every final-test number.
- Any mic sample rate is accepted and resampled to 16 kHz before mel
  extraction (browsers commonly record at 44.1/48 kHz).

## Checkpoints available in the dropdown

All 8 final-test checkpoints (2 teachers x 4 KD methods), so the same
recording can be compared across all of them without re-recording:

| Teacher | Methods |
| --- | --- |
| Qwen2.5-Omni (multimodal, prompted) | CE-only, Logit KD, Feature KD, Full KD |
| HuBERT-large-ll60k (audio-only, frozen) | CE-only, Logit KD, Feature KD, Full KD |

## Run it

```bash
python src/fsc/live_demo/app.py
```

Open the printed local URL (default `http://127.0.0.1:7860`) in a browser
and allow microphone access -- `localhost` is treated as a secure context by
browsers, so no HTTPS/certificate setup is required.

Record a command, check the predicted top-5 intents, then switch the
checkpoint dropdown to see how a different teacher/method classifies the
**same** recording without needing to speak again.

Requires `gradio` (installed separately -- not yet pinned in
`requirements.txt`; add `gradio>=6.0` there if you want it to persist
across environment rebuilds).
