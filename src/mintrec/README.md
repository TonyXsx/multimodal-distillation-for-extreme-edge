# MIntRec 2.0 pipeline (multimodal: audio + visual + text)

Next stage. Mirror the FSC pipeline structure, reusing `src/common/`:
  data_prep.py        — build MIntRec subsets
  extract_features.py — Qwen2.5-Omni teacher over audio+video+text -> hidden states
  train_probe.py      — bottleneck probe (reuse common.probe.Probe)
  student/            — student model + KD training (reuse common.losses / training / augment)

Paths: use common.config (DATA_ROOT/"mintrec", OUTPUTS_ROOT/"mintrec"). Data lives
under data/mintrec/ (physically on E: via the junction).
