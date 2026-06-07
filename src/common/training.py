"""Dataset-agnostic training/eval utilities."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, X, y, batch_size=256, device=None):
    """Batched eval of a (z, logits)-returning model. Returns acc / macro-F1 / weighted-F1."""
    device = device or DEVICE
    model.eval()
    preds = []
    for i in range(0, X.shape[0], batch_size):
        _, logits = model(X[i:i + batch_size].to(device))
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    yt = y.numpy()
    return {
        "acc": accuracy_score(yt, preds),
        "macro_f1": f1_score(yt, preds, average="macro"),
        "weighted_f1": f1_score(yt, preds, average="weighted"),
    }


def stratified_indices(labels, frac, seed):
    """Sorted indices of a stratified `frac` subset of `labels`, RNG-seeded by `seed`."""
    rng = np.random.RandomState(seed)
    y = labels.numpy()
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        k = max(1, int(round(len(c_idx) * frac)))
        idx.extend(rng.choice(c_idx, size=k, replace=False))
    return np.sort(np.array(idx))
