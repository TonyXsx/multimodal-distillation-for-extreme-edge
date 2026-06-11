# MIntRec 2.0 — frozen multimodal-teacher audio probe

Quick validation of the **privileged / cross-modal distillation** setup before
committing to student KD: the teacher sees all three modalities, but we probe
only the **audio-token** hidden states (what an audio-only student must mimic).

## Design (decided from the FSC results)

- **Teacher**: Qwen2.5-Omni-3B, 4-bit NF4, frozen.
- **Input** (`prompt_first`, single forward pass):
  `[ text (task prompt + transcript) ] + [ video frames ] + [ audio ]` — **audio last**,
  so under causal attention the audio tokens absorb the text+video context.
  Video frames are fed *without* their audio track (`use_audio_in_video=False`)
  and the clip audio is a separate `audio` part → one clean audio block at the end.
- **Pooling**: `audio_mean` over the audio-token block (the FSC winner), at layers
  `[24, 27, 30, 34]` plus their mean `pf_audio_mean_L24-27-30-34`.
- **Probe**: linear (A1) + 1024-MLP upper bound (A2); train on `train`, eval on `dev`,
  standardize with train stats, fixed 50-epoch schedule (no selection on dev).

Everything except the teacher-modality choice is frozen from the FSC ablation, so
this is ~zero new hyperparameter search.

## Run

```bash
# 1) extract audio hidden states for train + dev (resume-safe, sharded)
python src/mintrec/teacher_probe/extract_features.py --split all
#    smoke test first:  --split dev --limit 50

# 2) linear probe on dev
python src/mintrec/teacher_probe/train_probe.py
```

## Outputs

- `data/teacher_features/<FEAT_TAG>/{train,dev}_features.pt` + `extraction_config.json`
- `outputs/mintrec/teacher_probe/results.csv` — dev acc / macro-F1 per feature × arch

## Reading the result

The dev accuracy is the **go/no-go** signal:
- **high** → frozen multimodal teacher is a strong KD target → proceed to the
  audio-only student.
- **low** → motivates a teacher-modality ablation (audio-only vs +video vs +text
  targets) and/or LoRA teacher adaptation.

Note: the teacher's *generated* zero-shot label accuracy (see the early-experiment
notebook) is poor and irrelevant — the probe on hidden states is the real metric.
