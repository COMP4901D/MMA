"""
Centaur-style Convolutional Denoising Autoencoder (DAE) for sensor data cleaning.

Reference:
    Xaviar et al., "Centaur: Robust Multimodal Fusion for HAR",
    IEEE Sensors Journal, 2024. (arXiv: 2303.04636)

The cleaning module is trained independently from the HAR model:
  - Input:  corrupted sensor data  (any of Centaur's 4 corruption modes)
  - Target: original clean sensor data
  - Loss:   MSE(reconstructed, clean)

Two DAEs are provided:
  - IMU_DAE:  1-D convolutional autoencoder for IMU time-series (B, T, C)
  - RGBD_DAE: 2-D fully-convolutional autoencoder applied per-frame (B, T, 4, H, W)
"""

import torch
import torch.nn as nn


class IMU_DAE(nn.Module):
    """1-D Convolutional Denoising Autoencoder for IMU signals.

    Adapted from Centaur's Conv2D DAE to Conv1D since IMU has only 6 channels
    (too narrow for spatial 2D convolution).

    Input:  (B, T, C)  e.g. (B, 128, 6)
    Output: (B, T, C)  reconstructed clean signal
    """

    def __init__(self, in_channels: int = 6, seq_len: int = 128,
                 latent_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        # Encoder: 3 Conv1d layers (channels: in_channels -> 32 -> 64 -> 128)
        # Kernel=5, stride=2, padding=2  =>  T -> T//2 each layer
        # 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # After encoder: (B, 128, seq_len//8)
        enc_len = seq_len // 8  # 128 // 8 = 16
        self.flatten_dim = 128 * enc_len

        # Bottleneck
        self.fc_encode = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder: 3 ConvTranspose1d layers (reverse of encoder)
        self.dec_reshape_c = 128
        self.dec_reshape_l = enc_len

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, in_channels, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            # No activation — IMU is z-scored (not bounded [0,1])
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C) corrupted IMU signal
        Returns:
            (B, T, C) reconstructed clean signal
        """
        # (B, T, C) -> (B, C, T) for Conv1d
        h = x.transpose(1, 2)

        # Encode
        h = self.encoder(h)                         # (B, 128, T//8)
        h = h.reshape(h.size(0), -1)                # (B, flatten_dim)
        z = self.fc_encode(h)                        # (B, latent_dim)

        # Decode
        h = self.fc_decode(z)                        # (B, flatten_dim)
        h = h.reshape(-1, self.dec_reshape_c,
                       self.dec_reshape_l)            # (B, 128, T//8)
        h = self.decoder(h)                          # (B, C, T)

        # (B, C, T) -> (B, T, C)
        return h.transpose(1, 2)


class RGBD_DAE(nn.Module):
    """2-D Fully-Convolutional Denoising Autoencoder for RGBD frames.

    Applied independently to each frame (no temporal context within the DAE).
    Lightweight to avoid overfitting on the small UTD-MHAD dataset.

    Input:  (B, T, 4, H, W)  e.g. (B, 16, 4, 112, 112)
    Output: (B, T, 4, H, W)  reconstructed clean frames
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # Encoder: 3 layers  (4 -> 32 -> 64 -> 128)
        # 112 -> 56 -> 28 -> 14
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

        # Decoder: 3 layers (reverse)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.Sigmoid(),  # RGBD values are in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) corrupted RGBD frames
        Returns:
            (B, T, C, H, W) reconstructed clean frames
        """
        B, T, C, H, W = x.shape
        # Flatten batch and time -> per-frame processing
        h = x.reshape(B * T, C, H, W)
        h = self.encoder(h)
        h = self.decoder(h)
        return h.reshape(B, T, C, H, W)


class CleaningDAE(nn.Module):
    """Combined cleaning module for both RGBD and IMU modalities.

    At inference time:
        corrupted (rgbd, imu) -> CleaningDAE -> cleaned (rgbd, imu) -> HAR model
    """

    def __init__(self, imu_channels: int = 6, imu_seq_len: int = 128,
                 imu_latent_dim: int = 64,
                 rgbd_channels: int = 4,
                 enable_imu: bool = True,
                 enable_rgbd: bool = True):
        super().__init__()
        self.enable_imu = enable_imu
        self.enable_rgbd = enable_rgbd

        if enable_imu:
            self.imu_dae = IMU_DAE(imu_channels, imu_seq_len, imu_latent_dim)
        if enable_rgbd:
            self.rgbd_dae = RGBD_DAE(rgbd_channels)

    def forward(self, rgbd, imu):
        """Clean both modalities.

        Args:
            rgbd: (B, T, C, H, W) or None
            imu:  (B, T, C) or None
        Returns:
            cleaned (rgbd, imu) tuple
        """
        if self.enable_rgbd and rgbd is not None:
            rgbd = self.rgbd_dae(rgbd)
        if self.enable_imu and imu is not None:
            imu = self.imu_dae(imu)
        return rgbd, imu
