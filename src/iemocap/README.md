# IEMOCAP pipeline (multimodal: audio + visual + text)

Mirrors the FSC / MIntRec structure, reusing `src/common/`:

```
teacher_probe/
    extract_features_iemocap.ipynb  — exploratory: try the Qwen2.5-Omni teacher
                                       extraction LOCALLY (4-bit, heavily sub-sampled
                                       video frames) before committing to a full run
    extract_features.py             — (later) sharded/resume-safe port of the notebook
    train_probe.py                  — (later) bottleneck probe (reuse common.probe.Probe)
student/                            — (later) audio(+visual) student + KD
```

## Task

IEMOCAP standard **4-class single-label** emotion recognition:
`angry / happy(+excited) / sad / neutral` (the de-facto benchmark; `exc` is merged
into `hap`, all other categories and no-agreement `xxx` are dropped). Single-label
⇒ the FSC/MIntRec KD recipe (softmax + temperature) transfers unchanged.

## Teacher setup (kept ~identical to MIntRec)

- **Teacher**: Qwen2.5-Omni-3B, 4-bit NF4, frozen, `sdpa` attention.
- **Input** (`prompt_first`, single forward pass):
  `[ text (task prompt + transcript) ] + [ video frames ] + [ audio ]` — **audio last**,
  so under causal attention the audio tokens absorb the text+video context.
- **Video = heavily sub-sampled frames only.** IEMOCAP video is at the *dialog*
  level (a ~5-min two-person shot), so per utterance we seek into the dialog `.avi`
  and sample a few frames inside `[start, end]` (default 4, even-count required by
  the Qwen temporal patch=2; resized small to cap vision tokens). This matches the
  extreme-edge constraint (student sees ≤ a few frames) *and* keeps it runnable on a
  local 3060.
- **Audio**: the per-utterance `sentences/wav/<dialog>/<utt>.wav` (clean, already
  segmented).
- **Pooling**: `audio_mean` over the audio-token block at layers `[24, 27, 30, 34]`
  + their mean — the FSC/MIntRec winner.

## Data

Licence-gated; **cannot be auto-downloaded**. Request access from USC SAIL and place
the release here (data/ is junctioned to E:):

```
data/iemocap/IEMOCAP_full_release/
    Session1..5/
        dialog/EmoEvaluation/*.txt      — utterance emotion labels + [V,A,D] + times
        dialog/transcriptions/*.txt     — utterance transcripts (the 'text' modality)
        dialog/avi/DivX/*.avi           — dialog-level video (we frame-sample from it)
        sentences/wav/<dialog>/*.wav    — per-utterance audio
```

## Run

Open `teacher_probe/extract_features_iemocap.ipynb` and run top-to-bottom:
1. build the utterance index from `EmoEvaluation`,
2. **visualise sampled frames** for one utterance (decide `SPEAKER_CROP` =
   full/left/right by eye — the dialog shot may contain both actors),
3. load the 4-bit teacher and run a 5-sample smoke test (prints seq-len, #audio
   tokens, peak VRAM),
4. a tiny `--limit`-style extraction that saves a smoke `.pt`.

Once it runs cleanly, port the loop into `extract_features.py` (sharded, resume-safe)
exactly like `src/mintrec/teacher_probe/extract_features.py`.
