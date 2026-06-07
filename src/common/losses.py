"""Knowledge-distillation losses (dataset-agnostic)."""

import torch.nn.functional as F


def kd_logit_loss(student_logits, teacher_logits, t):
    """Temperature-scaled KL distillation loss (Hinton et al.)."""
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def kd_feature_loss(student_z, teacher_z):
    """Cosine-distance feature distillation (1 - mean cosine similarity)."""
    return 1.0 - F.cosine_similarity(student_z, teacher_z, dim=1).mean()
