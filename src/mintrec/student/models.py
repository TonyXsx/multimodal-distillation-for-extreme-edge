"""
MIntRec2.0 tiny students: audio-only and audio-visual.

Reuses the FSC audio encoder UNCHANGED (common.models.audio_student.DSResNetSE,
SMALL_KW — 97,991 params, ~0.37 MiB FP32). Longer MIntRec clips need no
architecture change: DSResNetSE global-average-pools over the full time-freq
map before the projection head, so param count is independent of input
length.

The visual encoder is built from the SAME primitives (SEBlock, DWSepConv,
ResDSSEBlock) as the audio encoder, just with a much smaller channel plan and
an RGB stem, so both branches share one "small, depthwise-separable + SE"
design language. Per-frame features are mean-pooled over the (small, fixed)
frame count -> one clip-level visual embedding.

Budget (see README.md): audio-only must stay near FSC's ~0.37 MiB; the
audio-visual student's total should stay under ~1 MiB (262,144 FP32 params),
with a fallback ceiling of ~3 MiB (786,432 params) if 1 MiB proves too small
to be useful. Run `python -m mintrec.student.models` to print exact counts.
"""

import torch
import torch.nn as nn

from common.models.audio_student import DSResNetSE, SEBlock, DWSepConv, ResDSSEBlock, model_summary

AUDIO_KW = {"channels": (16, 32, 64, 96, 128), "proj_hidden": None}   # FSC's SMALL_KW, unchanged
PROJ_DIM = 64                                                         # shared bottleneck dim (Feature-KD target dim)
N_CLASSES = 30


class VisualEncoder(nn.Module):
    """Tiny per-frame CNN (same block primitives as the audio encoder) + mean-pool over frames.

    Input:  [B, F, 3, H, W]  (F = frame count, e.g. 8; H=W=64)
    Output: z_visual [B, proj_dim]
    """

    def __init__(self, channels=(6, 12, 20, 28), proj_dim=PROJ_DIM, dropout=0.2, se_r=4):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResDSSEBlock(c0, c1, stride=2, r=se_r)
        self.block2 = ResDSSEBlock(c1, c2, stride=2, r=se_r)
        self.block3 = ResDSSEBlock(c2, c3, stride=2, r=se_r)
        self.proj = nn.Linear(c3, proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, F, 3, H, W] -> fold frames into batch for a shared per-frame CNN
        b, f = x.shape[:2]
        x = x.view(b * f, *x.shape[2:])
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=(2, 3))                 # [B*F, c3] global average pool
        x = self.dropout(x)
        x = self.proj(x)                       # [B*F, proj_dim]
        x = x.view(b, f, -1).mean(dim=1)        # mean-pool over frames -> [B, proj_dim]
        return x


class AudioOnlyStudent(nn.Module):
    """Thin wrapper around the unmodified FSC small DSResNet-SE (audio-only baseline)."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.audio = DSResNetSE(n_classes=n_classes, **AUDIO_KW)

    def forward(self, logmel, frames=None):
        z, logits = self.audio(logmel)
        return z, logits


class AudioVisualStudent(nn.Module):
    """Audio encoder (unchanged FSC small student, minus its own classifier) +
    tiny visual encoder -> concat -> fusion MLP -> proj_dim bottleneck -> classifier."""

    def __init__(self, n_classes=N_CLASSES, visual_channels=(6, 12, 20, 28),
                 proj_dim=PROJ_DIM, dropout=0.2):
        super().__init__()
        self.audio = DSResNetSE(n_classes=n_classes, **AUDIO_KW)   # reuse whole module; its own
                                                                     # (z, logits) head is unused here,
                                                                     # only its trunk + bottleneck_norm(proj) matter
        self.visual = VisualEncoder(channels=visual_channels, proj_dim=proj_dim, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.bottleneck_norm = nn.BatchNorm1d(proj_dim)
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, logmel, frames):
        z_a, _ = self.audio(logmel)             # [B, proj_dim] audio bottleneck (its own logits unused)
        z_v = self.visual(frames)                # [B, proj_dim]
        z = self.fusion(torch.cat([z_a, z_v], dim=1))
        z = self.bottleneck_norm(z)
        logits = self.classifier(z)
        return z, logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    ao = AudioOnlyStudent()
    av = AudioVisualStudent()
    for name, m in [("audio-only", ao), ("audio-visual", av)]:
        n = count_params(m)
        print(f"[{name}] params={n:,}  FP32={n*4/1024**2:.3f} MiB  INT8(est)={n/1024**2:.3f} MiB")

    logmel = torch.randn(2, 1, 601, 64)
    frames = torch.randn(2, 8, 3, 64, 64)
    z, logits = ao(logmel)
    print("audio-only  ->", z.shape, logits.shape)
    z, logits = av(logmel, frames)
    print("audio-visual ->", z.shape, logits.shape)
