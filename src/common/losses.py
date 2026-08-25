"""KD losses."""

import torch
import torch.nn.functional as F


def kd_logit_loss(student_logits, teacher_logits, t):
    """Hinton KD, KL between temperature-softened logits."""
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def kd_feature_loss(student_z, teacher_z):
    """1 - mean cosine sim."""
    return 1.0 - F.cosine_similarity(student_z, teacher_z, dim=1).mean()


def _pdist(e, eps=1e-12):
    # pairwise distances in a batch, diagonal zeroed
    sq = e.pow(2).sum(dim=1)
    d = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (e @ e.t())).clamp(min=eps).sqrt()
    return d - torch.diag_embed(torch.diagonal(d))


def rkd_distance_loss(student_z, teacher_z):
    """Distance-wise RKD (Park et al. 2019).

    Each distance matrix is divided by its own mean, so only the ratios matter
    and a global rescaling of either embedding doesn't change the loss. Worth
    trying because about 42% of each teacher target vector points along one
    shared direction, and the cosine loss makes the student copy that.
    """
    with torch.no_grad():
        td = _pdist(teacher_z)
        td = td / (td[td > 0].mean() + 1e-12)
    sd = _pdist(student_z)
    sd = sd / (sd[sd > 0].mean() + 1e-12)
    return F.smooth_l1_loss(sd, td)


def rkd_angle_loss(student_z, teacher_z):
    """Angle-wise RKD: cos of the angle at every vertex of every triplet in the
    batch. Also invariant to shift and scale."""
    def angles(e):
        v = F.normalize(e.unsqueeze(0) - e.unsqueeze(1), p=2, dim=2)
        return torch.bmm(v, v.transpose(1, 2)).view(-1)

    with torch.no_grad():
        ta = angles(teacher_z)
    return F.smooth_l1_loss(angles(student_z), ta)


def rkd_loss(student_z, teacher_z, w_dist=1.0, w_angle=2.0):
    """1:2 distance-to-angle weighting, same as the paper."""
    return w_dist * rkd_distance_loss(student_z, teacher_z) + \
        w_angle * rkd_angle_loss(student_z, teacher_z)
