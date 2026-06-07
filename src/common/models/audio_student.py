"""
DSResNet-SE: compact audio-only student.

Pipeline:
    log-mel [B,1,T,64]
      -> Conv2D stem (1->c0, stride (2,2))
      -> 4x ResDS-SE blocks (stem/b1-2 stride (2,2); b3-4 stride (2,1), so the
         frequency axis stays at ~8 bins, not collapsed to 2 — better
         speaker-independent generalization)
      -> global average pooling
      -> projection head (c4 -> [proj_hidden ->] proj_dim, BatchNorm-normalized)
      -> n_classes classifier

forward() returns (student_z, student_logits) for KD. Parametrized by `channels`
so the same class serves the strong (default) and small students; proj_dim stays
64 so the Feature-KD target dim (teacher bottleneck = 64) is matched.

Each ResDS-SE block:
    x -> dwconv3x3 -> pwconv1x1 -> BN -> ReLU
      -> dwconv3x3 -> pwconv1x1 -> BN -> SE -> (+ residual) -> ReLU
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (reduction r)."""

    def __init__(self, channels, r=8):
        super().__init__()
        hidden = max(1, channels // r)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)      # squeeze (global avg pool)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s                               # excite


class DWSepConv(nn.Module):
    """Depthwise 3x3 then pointwise 1x1 (no activation/BN inside)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResDSSEBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, r=8):
        super().__init__()
        self.conv1 = DWSepConv(cin, cout, stride=stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = DWSepConv(cout, cout, stride=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SEBlock(cout, r)
        self.relu = nn.ReLU(inplace=True)

        if cin != cout or stride != 1:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.short = nn.Identity()

    def forward(self, x):
        identity = self.short(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + identity)


class DSResNetSE(nn.Module):
    """channels = (stem_out, b1, b2, b3, b4); proj_hidden=None -> project c4->proj_dim directly."""

    def __init__(self, n_mels=64, n_classes=31, channels=(32, 64, 128, 192, 256),
                 proj_hidden=128, proj_dim=64, dropout=0.2, se_r=8):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResDSSEBlock(c0, c1, stride=(2, 2), r=se_r)
        self.block2 = ResDSSEBlock(c1, c2, stride=(2, 2), r=se_r)
        self.block3 = ResDSSEBlock(c2, c3, stride=(2, 1), r=se_r)
        self.block4 = ResDSSEBlock(c3, c4, stride=(2, 1), r=se_r)

        if proj_hidden:
            self.proj = nn.Sequential(
                nn.Linear(c4, proj_hidden),
                nn.BatchNorm1d(proj_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(proj_hidden, proj_dim),
            )
        else:
            self.proj = nn.Linear(c4, proj_dim)           # direct head, e.g. 128 -> 64

        self.bottleneck_norm = nn.BatchNorm1d(proj_dim)   # normalize bottleneck
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        # x: [B, 1, T, n_mels]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=(2, 3))                      # global average pooling -> [B, c4]
        z = self.bottleneck_norm(self.proj(x))     # normalized bottleneck -> [B, proj_dim]
        logits = self.classifier(z)                # [B, n_classes]
        return z, logits


def model_summary(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": n_params,
            "fp32_mb": n_params * 4 / 1024 ** 2,
            "fp16_mb": n_params * 2 / 1024 ** 2,
            "int8_mb": n_params * 1 / 1024 ** 2}


if __name__ == "__main__":
    for name, kw in [("STRONG (default)", {}),
                     ("SMALL", {"channels": (16, 32, 64, 96, 128), "proj_hidden": None})]:
        m = DSResNetSE(**kw)
        s = model_summary(m)
        z, logits = m(torch.randn(2, 1, 301, 64))
        print(f"[{name}] params {s['params']:,}  FP32 {s['fp32_mb']:.2f} MB  "
              f"INT8(est) {s['int8_mb']:.2f} MB  -> z {tuple(z.shape)} logits {tuple(logits.shape)}")
