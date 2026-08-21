"""
MS-SENet (ICASSP 2024) -- PyTorch port of the official Keras implementation.

Paper : "MS-SENet: Enhancing Speech Emotion Recognition Through Multi-scale
         Feature Fusion with Squeeze-and-Excitation Blocks", arXiv:2312.11974
Code  : https://github.com/MengboLi/MS-SENet  (MS-SENet.py, TensorFlow/Keras)
Base  : TIM-Net, https://github.com/Jiaxin-Ye/TIM-Net_SER (ICASSP 2023)

The official repository is treated as authoritative and this is a faithful
port of it, not a reinterpretation. Where the common textual description of
the architecture disagrees with that code, the code is followed and the
difference is recorded here:

1. KERNEL ORIENTATION. The three frontend branches are Conv2D(39, (11,1)),
   (1,9) and (3,3) -- not (9,1)/(1,11). Keras feeds [B, T, F, 1], so the
   kernel is (time, freq): 11 frames along TIME, 9 bins along FREQUENCY.
   Descriptions that give 9x1 / 1x11 have the two axes swapped.

2. CONCATENATION AXIS. The branches are concatenated on axis=2, the FREQUENCY
   axis, not on channels: three [B,T,F,39] tensors become [B,T,3F,39]. The
   channel count stays 39 throughout, which is why the SE block's Dense layers
   are hard-coded to 39 units.

3. SE HAS NO REDUCTION. Both SE projections are Dense(39): squeeze -> 39 ->
   39, i.e. reduction ratio 1. Textbook SE uses r=8 or 16.

4. THE FRONTEND IS NOT SHARED. It is instantiated twice -- once on the input,
   once on the time-reversed input -- with independent weights, before the
   forward and backward temporal branches.

5. A RAW-INPUT SKIP. After the 1x1 channel collapse the frontend output
   [B,T,3F] is concatenated with the untouched input [B,T,F], giving [B,T,4F]
   = [B,T,156] for F=39, which is what the temporal projection consumes.

6. THE TAB GATE IS MULTIPLICATIVE, NOT ADDITIVE. A Temporal-Aware Block ends
   `F_x = original_x * sigmoid(conv2_out)`; there is no residual `add`. The
   block attenuates its own input rather than adding to it.

Default hyperparameters are the official IEMOCAP settings: 39 filters, kernel
size 2, one stack, dilations 2^0..2^9 (ten TABs per direction), dropout 0.1,
SpatialDropout 0.2 in the frontend.

The forward pass returns `(z, logits)` -- the same contract as DSResNetSE --
so the existing training and evaluation code can drive either backbone. Note
that `z` here is 39-dimensional (the width the paper's fusion produces), not
the 64 used by the DSResNet-SE student, which matters when a feature-KD
target has to match it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock2d(nn.Module):
    """Official `se_module`: squeeze over (time, freq), excite over channels,
    with NO reduction -- Dense(C) -> ReLU -> Dense(C) -> sigmoid."""

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):                       # [B, C, T, F]
        w = x.mean(dim=(2, 3))                  # squeeze -> [B, C]
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w[:, :, None, None]


class MultiScaleFrontend(nn.Module):
    """Three parallel Conv2d branches (11x1 time, 1x9 freq, 3x3 joint), each
    39 filters, BN + ReLU + SpatialDropout2d(0.2), concatenated along the
    FREQUENCY axis, SE-reweighted over channels, collapsed to one channel by a
    1x1 conv, then concatenated with the raw input.

    No pooling: the official code has its AveragePooling2D lines commented
    out, so temporal resolution is carried through intact.
    """

    def __init__(self, n_feat=39, filters=39, sd_rate=0.2):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(1, filters, (11, 1), padding="same"),   # along time
            nn.Conv2d(1, filters, (1, 9), padding="same"),    # along frequency
            nn.Conv2d(1, filters, (3, 3), padding="same"),    # joint
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(filters) for _ in range(3)])
        self.sdrop = nn.Dropout2d(sd_rate)
        self.se = SEBlock2d(filters)
        self.collapse = nn.Conv2d(filters, 1, 1)
        self.out_dim = 3 * n_feat + n_feat

    def forward(self, x):                       # x: [B, T, F]
        h = x.unsqueeze(1)                      # -> [B, 1, T, F]
        paths = [self.sdrop(F.relu(bn(conv(h))))
                 for conv, bn in zip(self.branches, self.bns)]
        h = torch.cat(paths, dim=3)             # concat on FREQUENCY -> [B, 39, T, 3F]
        h = self.se(h)
        h = F.relu(self.collapse(h)).squeeze(1)  # -> [B, T, 3F]
        return torch.cat([h, x], dim=2)         # raw-input skip -> [B, T, 4F]


class TemporalAwareBlock(nn.Module):
    """Two dilated causal convs, then a sigmoid gate applied multiplicatively
    to the block's own input (official: `F_x = original_x * sigmoid(out)`)."""

    def __init__(self, channels, kernel_size=2, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.sdrop = nn.Dropout1d(dropout)

    def _causal(self, conv, bn, x):
        return self.sdrop(F.relu(bn(conv(F.pad(x, (self.pad, 0))))))

    def forward(self, x):                       # [B, C, T]
        h = self._causal(self.conv1, self.bn1, x)
        h = self._causal(self.conv2, self.bn2, h)
        return x * torch.sigmoid(h)


class TemporalBranch(nn.Module):
    """1x1 projection followed by `n_tabs` Temporal-Aware Blocks with dilations
    1, 2, 4, ... 2^(n_tabs-1)."""

    def __init__(self, in_dim, filters=39, kernel_size=2, n_tabs=10, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, filters, 1)
        self.tabs = nn.ModuleList([
            TemporalAwareBlock(filters, kernel_size, 2 ** j, dropout) for j in range(n_tabs)
        ])

    def forward(self, x):                       # [B, T, in_dim]
        h = self.proj(x.transpose(1, 2))        # -> [B, filters, T]
        outs = []
        for tab in self.tabs:
            h = tab(h)
            outs.append(h)
        return outs                             # n_tabs x [B, filters, T]


class MSSENet(nn.Module):
    """MS-SENet with the official IEMOCAP configuration.

    Forward and backward branches each get their own frontend and their own
    TABs. At every dilation level the two are summed, globally average-pooled
    over time, and stacked; a learnable weight vector over the levels
    (`WeightLayer` in the official code -- a plain weighted sum, no softmax)
    fuses them into one utterance vector.
    """

    def __init__(self, n_feat=39, n_classes=4, filters=39, kernel_size=2,
                 n_tabs=10, dropout=0.1, sd_rate=0.2):
        super().__init__()
        self.front_fwd = MultiScaleFrontend(n_feat, filters, sd_rate)
        self.front_bwd = MultiScaleFrontend(n_feat, filters, sd_rate)
        self.branch_fwd = TemporalBranch(self.front_fwd.out_dim, filters, kernel_size, n_tabs, dropout)
        self.branch_bwd = TemporalBranch(self.front_bwd.out_dim, filters, kernel_size, n_tabs, dropout)
        # Keras `uniform` initialiser is RandomUniform(-0.05, 0.05).
        self.level_weights = nn.Parameter(torch.empty(n_tabs).uniform_(-0.05, 0.05))
        self.classifier = nn.Linear(filters, n_classes)
        self.embed_dim = filters

    def forward(self, x):                       # [B, T, F] or [B, 1, T, F]
        if x.dim() == 4:
            x = x.squeeze(1)
        fwd = self.branch_fwd(self.front_fwd(x))
        bwd = self.branch_bwd(self.front_bwd(torch.flip(x, dims=[1])))
        levels = torch.stack([(f + b).mean(dim=2) for f, b in zip(fwd, bwd)], dim=1)
        z = (levels * self.level_weights[None, :, None]).sum(dim=1)   # [B, filters]
        return z, self.classifier(z)


if __name__ == "__main__":
    m = MSSENet()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    z, logits = m(torch.randn(2, 606, 39))
    print(f"MS-SENet  params {n:,}  {n * 4 / 1024**2:.2f} MiB fp32  {n / 1024**2:.2f} MiB int8")
    print(f"  z {tuple(z.shape)}   logits {tuple(logits.shape)}")
