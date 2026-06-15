# IEMOCAP — frozen multimodal-teacher audio probe (exploratory)

Same **privileged / cross-modal distillation** setup as MIntRec: the teacher sees
all three modalities but we pool only the **audio-token** hidden states (what an
audio-only — or audio-led AV — student must mimic).

`extract_features_iemocap.ipynb` is the first step: confirm the Qwen2.5-Omni-3B
teacher extraction **runs locally** (4-bit, sub-sampled video frames) on IEMOCAP
before writing the full sharded extractor.

## What the notebook does

1. **Index** — walk `Session1..5/dialog/EmoEvaluation/*.txt`, parse each utterance's
   `[start - end]`, turn-name, emotion code and `[V, A, D]`; keep the 4 benchmark
   classes (`exc→hap`), attach the transcript from `dialog/transcriptions/`.
2. **Frame sampling** — for each utterance, seek into the dialog `.avi` (cv2) and
   grab `NUM_VIDEO_FRAMES` frames inside `[start, end]`, optional left/right crop to
   the speaking actor, resized to cap vision tokens.
3. **Visual check** — display the sampled frames so you can set `SPEAKER_CROP`.
4. **Teacher** — load 4-bit Qwen2.5-Omni-3B, build `prompt_first` inputs
   `[text, video-frames, audio]` (audio last, `use_audio_in_video=False`), run
   `thinker(..., output_hidden_states=True)`, pool `audio_mean` at layers
   `[24,27,30,34]`.
5. **Smoke test + mini extraction** — 5-sample timing/VRAM, then a small `limit`
   run saved to a smoke `.pt` (same dict layout as the MIntRec extractor).

## Knobs (top config cell)

| name | default | note |
|------|---------|------|
| `DTYPE` | `4bit` | `bf16` on a big GPU |
| `NUM_VIDEO_FRAMES` | `4` | **must be even**; try `2` if VRAM-tight, `8` if plenty |
| `FRAME_MAX_SIDE` | `336` | resize longest side → ~27 vision tokens/frame |
| `SPEAKER_CROP` | `full` | `left`/`right` once you've eyeballed the frames |

## Reading the result

If the 5-sample smoke test runs within VRAM and the audio-token block is located
correctly (non-zero `num_audio_tokens`), the local 4-bit path is viable → port to a
sharded `extract_features.py` for the full corpus, then probe the pooled features
(reuse `common.probe.Probe`) as the go/no-go signal, exactly as in MIntRec.
