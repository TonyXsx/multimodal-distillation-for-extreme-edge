"""Knowledge-distillation losses (dataset-agnostic)."""

import torch
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


def _pdist(e, eps=1e-12):
    """Pairwise Euclidean distances within a batch, diagonal forced to zero."""
    sq = e.pow(2).sum(dim=1)
    d = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (e @ e.t())).clamp(min=eps).sqrt()
    return d - torch.diag_embed(torch.diagonal(d))


def rkd_distance_loss(student_z, teacher_z):
    """Distance-wise Relational KD (Park et al., 2019).

    Matches the RATIO structure of pairwise distances rather than absolute
    positions: each distance matrix is divided by its own mean before the
    comparison, so the loss is invariant to any global scaling of either
    embedding. That is the property that makes it worth trying here -- the
    cosine feature loss is dragged around by a shared constant direction the
    student has to spend capacity reproducing, and roughly 42% of each teacher
    target vector points along it.
    """
    with torch.no_grad():
        td = _pdist(teacher_z)
        td = td / (td[td > 0].mean() + 1e-12)
    sd = _pdist(student_z)
    sd = sd / (sd[sd > 0].mean() + 1e-12)
    return F.smooth_l1_loss(sd, td)


def rkd_angle_loss(student_z, teacher_z):
    """Angle-wise Relational KD: matches cos of the angle at every vertex of
    every triplet in the batch. A higher-order relation than distance, and
    likewise invariant to translation and scale of the embedding."""
    def angles(e):
        v = F.normalize(e.unsqueeze(0) - e.unsqueeze(1), p=2, dim=2)
        return torch.bmm(v, v.transpose(1, 2)).view(-1)

    with torch.no_grad():
        ta = angles(teacher_z)
    return F.smooth_l1_loss(angles(student_z), ta)


def rkd_loss(student_z, teacher_z, w_dist=1.0, w_angle=2.0):
    """Combined RKD. The 1:2 distance-to-angle weighting is the paper's."""
    return w_dist * rkd_distance_loss(student_z, teacher_z) + \
        w_angle * rkd_angle_loss(student_z, teacher_z)
