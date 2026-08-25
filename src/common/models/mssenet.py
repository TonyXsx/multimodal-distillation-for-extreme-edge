"""
MS-SENet (ICASSP 2024), ported from the official Keras code.

paper: arXiv:2312.11974
code:  https://github.com/MengboLi/MS-SENet (MS-SENet.py)
base:  TIM-Net, https://github.com/Jiaxin-Ye/TIM-Net_SER

I followed the repo rather than the paper text where the two disagree. Things
that surprised me while porting, written down so I don't re-derive them:

1. Kernel orientation. The frontend branches are (11,1), (1,9) and (3,3).
   Keras feeds [B,T,F,1] so the kernel is (time, freq), i.e. 11 frames along
   time and 9 bins along frequency. Descriptions giving 9x1 / 1x11 have the
   axes the wrong way round.

2. Concat axis is 2, the frequency axis, not channels. Three [B,T,F,39]
   become [B,T,3F,39]. Channels stay 39 the whole way, which is why the SE
   Dense layers are hardcoded to 39.

3. SE has no reduction. Both projections are Dense(39), so r=1. Usual SE uses
   8 or 16.

4. The frontend is not shared. It gets built twice, once on the input and once
   on the reversed input, with separate weights.

5. There is a raw-input skip. After the 1x1 collapse, [B,T,3F] is concatenated
   with the untouched input, giving [B,T,4F] = [B,T,156] for F=39.

6. The TAB gate multiplies, it doesn't add. The block ends with
   `F_x = original_x * sigmoid(conv2_out)`, no residual add.

Defaults are the official IEMOCAP ones: 39 filters, kernel 2, one stack,
dilations 2^0..2^9, dropout 0.1, spatial dropout 0.2.

forward() returns (z, logits) like DSResNetSE so the same training code drives
both. z is 39-d here, not 64, which matters if you want a feature-KD target to
line up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock2d(nn.Module):
    """squeeze over (time, freq), excite over channels. no reduction."""

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):                       # [B, C, T, F]
        w = x.mean(dim=(2, 3))                  # squeeze -> [B, C]
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w[:, :, None, None]


class MultiScaleFrontend(nn.Module):
    """Three parallel conv branches (11x1 time, 1x9 freq, 3x3 joint), 39 filters
    each, concatenated along frequency, SE over channels, collapsed to one
    channel by a 1x1, then concatenated with the raw input.

    No pooling. The official code has its AveragePooling2D commented out, so
    time resolution goes through untouched.
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
    """two dilated causal convs, then a sigmoid gate multiplied onto the input."""

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
    """1x1 projection then n_tabs TABs, dilations 1, 2, 4, ..."""

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
    """MS-SENet, official IEMOCAP config.

    Forward and backward branches get their own frontend and their own TABs. At
    each dilation level the two are summed, pooled over time and stacked, then
    a learnable weight vector over levels fuses them into one vector. That is
    `WeightLayer` in the official code, just a weighted sum, no softmax.
    """

    def __init__(self, n_feat=39, n_classes=4, filters=39, kernel_size=2,
                 n_tabs=10, dropout=0.1, sd_rate=0.2):
        super().__init__()
        self.front_fwd = MultiScaleFrontend(n_feat, filters, sd_rate)
        self.front_bwd = MultiScaleFrontend(n_feat, filters, sd_rate)
        self.branch_fwd = TemporalBranch(self.front_fwd.out_dim, filters, kernel_size, n_tabs, dropout)
        self.branch_bwd = TemporalBranch(self.front_bwd.out_dim, filters, kernel_size, n_tabs, dropout)
        # keras `uniform` is RandomUniform(-0.05, 0.05)
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
