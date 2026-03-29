"""SpatialCNN backbone for per-frame RGBD feature extraction."""

import torch.nn as nn


class SpatialCNN(nn.Module):
    """
    Lightweight 2D CNN:  (B, 4, 112, 112) -> (B, d_model)
    4 conv blocks with stride-2 -> AdaptiveAvgPool -> Linear
    112 -> 56 -> 28 -> 14 -> 7 -> 1
    """

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(256, d_model)

    def forward(self, x):
        """x: (B, C, H, W) -> (B, d_model)"""
        return self.proj(self.features(x))
