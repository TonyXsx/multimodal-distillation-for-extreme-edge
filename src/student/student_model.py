"""
DSResNet-SE: compact audio-only student for FSC intent classification.

Pipeline (see student/README.md):
    log-mel [B,1,T,64]
      -> Conv2D stem (1->32, stride 2)
      -> 4x ResDS-SE blocks (32->64->128->192->256, each stride 2)
      -> global average pooling
      -> projection head (256->128->64)
      -> 31-way classifier

forward() returns (student_z_64, student_logits) for KD.

Each ResDS-SE block (per README):
    x -> dwconv3x3 -> pwconv1x1 -> BN -> ReLU
      -> dwconv3x3 -> pwconv1x1 -> BN -> SE -> (+ residual) -> ReLU
A 1x1 conv matches the residual path when channels/stride differ.

Design note: the README lists the block internals but not inter-block
downsampling. Stride-2 is applied in the stem and in each block's first
depthwise conv (standard for efficient audio CNNs, e.g. BC-ResNet) so that
activation memory is feasible on a 6 GB GPU. This does not change the
parameter count (~0.37M), which stays in the README's 0.35-0.45M target.
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
    def __init__(self, n_mels=64, n_classes=31, proj_dim=64, dropout=0.1, se_r=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResDSSEBlock(32, 64, stride=2, r=se_r)
        self.block2 = ResDSSEBlock(64, 128, stride=2, r=se_r)
        self.block3 = ResDSSEBlock(128, 192, stride=2, r=se_r)
        self.block4 = ResDSSEBlock(192, 256, stride=2, r=se_r)

        self.proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        # x: [B, 1, T, n_mels]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=(2, 3))          # global average pooling -> [B, 256]
        z = self.proj(x)                # student bottleneck -> [B, proj_dim]
        logits = self.classifier(z)     # [B, n_classes]
        return z, logits


def model_summary(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_mb = n_params * 4 / 1024 ** 2
    fp16_mb = n_params * 2 / 1024 ** 2
    int8_mb = n_params * 1 / 1024 ** 2
    return {"params": n_params, "fp32_mb": fp32_mb, "fp16_mb": fp16_mb, "int8_mb": int8_mb}


if __name__ == "__main__":
    m = DSResNetSE()
    s = model_summary(m)
    print(f"Trainable params : {s['params']:,}")
    print(f"FP32 size        : {s['fp32_mb']:.2f} MB")
    print(f"FP16 size        : {s['fp16_mb']:.2f} MB")
    print(f"INT8 (est) size  : {s['int8_mb']:.2f} MB")
    x = torch.randn(2, 1, 301, 64)
    z, logits = m(x)
    print(f"input  {tuple(x.shape)} -> z {tuple(z.shape)}  logits {tuple(logits.shape)}")
