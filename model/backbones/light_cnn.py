"""LightCNN backbone with SE attention for Depth / GAF images."""

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation attention."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        mid = max(ch // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out * self.se(out).unsqueeze(-1).unsqueeze(-1)
        return F.relu(out + x, inplace=True)


class LightCNN(nn.Module):
    """
    Lightweight 2D CNN + SE residual blocks.
    Used for Depth frames and IMU-GAF images.
    """

    def __init__(self, in_ch: int, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True),
            ResBlock(32),
            nn.MaxPool2d(2),
            # Stage 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            ResBlock(64),
            nn.MaxPool2d(2),
            # Stage 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            ResBlock(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.05),
            # Stage 4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            ResBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        """(B, C, H, W) -> (B, feat_dim)"""
        return self.fc(self.net(x).flatten(1))
