# FSC Hidden Representation Probing Plan

## Goal

We want to quickly test which hidden representations from Qwen2.5-Omni are useful for FSC intent classification. The first stage should focus on extracting and saving general pooled hidden features from the teacher model. After that, we can repeatedly train linear probes locally without rerunning the expensive teacher forward pass.

The goal is not to test every possible hidden state. The goal is to store a compact but informative set of representations that allows us to compare:

* projected audio embeddings before the LLM blocks
* audio-token hidden states under `prompt_first`
* task-aware text/prompt hidden states under `audio_first`
* different middle, late, and final layers
* single-layer features and later multi-layer combinations

---

## Layer Selection

The model has 36 transformer layers.

Use the following selected layers:

```python
selected_layers = [0, 9, 18, 24, 27, 30, 34, 36]
```

Here:

```python
layer 0 = projected audio embeddings before the LLM transformer blocks
layers 9, 18, 24, 27, 30, 34, 36 = hidden states after the corresponding LLM blocks
```

The actual LLM layers to extract are:

```python
llm_layers = [9, 18, 24, 27, 30, 34, 36]
```

Reasoning:

* layer 0 gives a baseline before any LLM processing
* layer 9 gives an early/mid reference
* layer 18 gives a middle-layer representation
* layers 24, 27, 30 focus on middle-late and late semantic representations
* layer 34 tests near-final representations
* layer 36 tests the final layer

This is intentionally biased toward later layers because the task is intent classification, so semantic and task-aware representations are more relevant than low-level acoustic representations.

---

## Prompt Orders

We should test two input orders.

### 1. `prompt_first`

```text
[prompt tokens] + [audio tokens]
```

In a causal decoder-only model, audio tokens can attend to the preceding prompt tokens. Therefore, audio-token representations in this setting may become more task-aware.

This setting is especially important for future representation-level KD, because the student will likely be audio-only and we may want it to match teacher-side audio-token representations.

### 2. `audio_first`

```text
[audio tokens] + [prompt tokens]
```

In this setting, the final text/prompt tokens can attend to both the audio tokens and the task instruction. Therefore, the last prompt/text hidden states are likely to be the strongest task-aware representations.

This setting is useful as a probing upper bound, although it may be less directly suitable as a distillation target for an audio-only student.

---

## Features to Save

Do not save full token-level hidden states for all samples. That would be too large.

Instead, save pooled features only.

For each sample, save the following.

---

## Layer 0 Features

Layer 0 corresponds to the projected audio embeddings before the LLM transformer blocks.

Save:

```text
projected_audio_mean
projected_audio_last
```

Definitions:

```text
projected_audio_mean = mean pooling over projected audio tokens
projected_audio_last = hidden vector of the last projected audio token
```

These features test whether the audio encoder/projector already contains linearly separable intent information before the LLM layers.

---

## `prompt_first` Features

Input order:

```text
[prompt tokens] + [audio tokens]
```

For each layer in:

```python
llm_layers = [9, 18, 24, 27, 30, 34, 36]
```

Save:

```text
prompt_first_L{L}_audio_mean
prompt_first_L{L}_audio_last
```

Definitions:

```text
audio_mean = mean pooling over all audio-token hidden states at layer L
audio_last = hidden vector of the last audio token at layer L
```

Example feature names:

```text
prompt_first_L9_audio_mean
prompt_first_L9_audio_last
prompt_first_L18_audio_mean
prompt_first_L18_audio_last
prompt_first_L24_audio_mean
prompt_first_L24_audio_last
prompt_first_L27_audio_mean
prompt_first_L27_audio_last
prompt_first_L30_audio_mean
prompt_first_L30_audio_last
prompt_first_L34_audio_mean
prompt_first_L34_audio_last
prompt_first_L36_audio_mean
prompt_first_L36_audio_last
```

These are the most important candidate features for future audio-only student distillation.

---

## `audio_first` Features

Input order:

```text
[audio tokens] + [prompt tokens]
```

For each layer in:

```python
llm_layers = [9, 18, 24, 27, 30, 34, 36]
```

Save:

```text
audio_first_L{L}_audio_mean
audio_first_L{L}_audio_last
audio_first_L{L}_last_text
audio_first_L{L}_last_4_text_mean
```

Definitions:

```text
audio_mean = mean pooling over all audio-token hidden states at layer L
audio_last = hidden vector of the last audio token at layer L
last_text = hidden vector of the final text/prompt token at layer L
last_4_text_mean = mean pooling over the final 4 text/prompt tokens at layer L
```

Example feature names:

```text
audio_first_L9_audio_mean
audio_first_L9_audio_last
audio_first_L9_last_text
audio_first_L9_last_4_text_mean

audio_first_L18_audio_mean
audio_first_L18_audio_last
audio_first_L18_last_text
audio_first_L18_last_4_text_mean

audio_first_L24_audio_mean
audio_first_L24_audio_last
audio_first_L24_last_text
audio_first_L24_last_4_text_mean

audio_first_L27_audio_mean
audio_first_L27_audio_last
audio_first_L27_last_text
audio_first_L27_last_4_text_mean

audio_first_L30_audio_mean
audio_first_L30_audio_last
audio_first_L30_last_text
audio_first_L30_last_4_text_mean

audio_first_L34_audio_mean
audio_first_L34_audio_last
audio_first_L34_last_text
audio_first_L34_last_4_text_mean

audio_first_L36_audio_mean
audio_first_L36_audio_last
audio_first_L36_last_text
audio_first_L36_last_4_text_mean
```

The `last_text` and `last_4_text_mean` features are expected to be strong task-aware representations because these text tokens can attend to both the audio and the task instruction.

The `audio_mean` and `audio_last` features are useful as controls. In `audio_first`, audio tokens cannot attend to the later prompt tokens because of causal masking, so comparing them with `prompt_first` audio-token features can show whether the prompt actually makes audio-token representations more task-aware.

---

## Total Number of Saved Feature Vectors Per Sample

Layer 0:

```text
2 vectors
```

`prompt_first`:

```text
7 layers × 2 features = 14 vectors
```

`audio_first`:

```text
7 layers × 4 features = 28 vectors
```

Total:

```text
2 + 14 + 28 = 44 vectors per sample
```

This is still manageable because each vector is only one pooled hidden representation, not the full token-level hidden state.

If hidden dimension is 2048 and features are stored in float16:

```text
44 × 2048 × 2 bytes ≈ 180 KB per sample
```

For 1000 samples:

```text
≈ 180 MB
```

For 5000 samples:

```text
≈ 900 MB
```

This is acceptable for local experiments.

---

## Suggested Storage Format

Save features as `.pt` files, for example:

```text
features/
  train_features.pt
  val_features.pt
  test_features.pt
```

Each file can be a dictionary:

```python
{
    "labels": labels,
    "sample_ids": sample_ids,
    "metadata": metadata,
    "features": {
        "projected_audio_mean": tensor,
        "projected_audio_last": tensor,

        "prompt_first_l9_audio_mean": tensor,
        "prompt_first_l9_audio_last": tensor,
        ...

        "audio_first_l9_audio_mean": tensor,
        "audio_first_l9_audio_last": tensor,
        "audio_first_l9_last_text": tensor,
        "audio_first_l9_last_4_text_mean": tensor,
        ...
    }
}
```

Each feature tensor should have shape:

```python
[num_samples, hidden_dim]
```

---

## Metadata to Save

For debugging, also save:

```text
sample_id
label
audio_path or original dataset index
speaker_id, if available
num_audio_tokens
num_text_tokens
audio_token_range
text_token_range
seq_len
```

The token ranges are important because hidden-state probing can easily go wrong if we accidentally pool over special tokens, prompt tokens, or padding tokens instead of the intended audio/text tokens.

---

## Feature Extraction Scope

At this stage, we only focus on extracting and saving hidden representations from the frozen teacher model.

No linear probing, classifier training, feature selection, or representation comparison experiments are included in this phase. The objective is simply to build a reusable feature bank so that all downstream analyses can be performed later without rerunning the expensive teacher forward pass.

The extraction pipeline should:

1. Run the frozen teacher model on the FSC dataset.
2. Extract hidden representations from the selected layers:

```python
selected_layers = [0, 9, 18, 24, 27, 30, 34, 36]
llm_layers = [9, 18, 24, 27, 30, 34, 36]
```

3. Process both prompt orders:

```text
prompt_first
audio_first
```

4. Compute the pooled representations defined earlier:

```text
layer 0:
- projected_audio_mean
- projected_audio_last

prompt_first:
- audio_mean
- audio_last

audio_first:
- audio_mean
- audio_last
- last_text
- last_4_text_mean
```

5. Save all extracted features together with labels and metadata.

---

## Summary

Use:

```python
selected_layers = [0, 9, 18, 24, 27, 30, 34, 36]
llm_layers = [9, 18, 24, 27, 30, 34, 36]
```

Save:

```text
layer 0:
- projected_audio_mean
- projected_audio_last

prompt_first:
- audio_mean
- audio_last

audio_first:
- audio_mean
- audio_last
- last_text
- last_4_text_mean
```

Total:

```text
44 pooled feature vectors per sample
```

The output of this stage is a compact feature bank extracted from the frozen teacher model, which can later be used for probing, representation analysis, or knowledge distillation experiments without requiring additional teacher inference.
