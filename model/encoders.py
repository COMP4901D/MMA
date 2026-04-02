"""
Modality-specific encoders.

RGBDEncoder: Spatial 2D CNN (per-frame) + MomentumMamba (temporal)
IMUEncoder:  Conv1D frontend  or  ConvNeXtV2 (GAF images) + MomentumMamba
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm
from .mamba import MomentumMambaBlock


# ---------------------------------------------------------------------------
#  Temporal denoising / smoothing filters
# ---------------------------------------------------------------------------

class TemporalDenoise(nn.Module):
    """Temporal denoising filter applied along the time axis.

    All modes accept (B, T, C) and return (B, T, C).

    Modes
    -----
    moving_avg : uniform 1-D depthwise conv (fixed)
    gaussian   : Gaussian-weighted 1-D depthwise conv (fixed)
    learnable  : depthwise conv initialised as Gaussian (trainable)
    ema        : bidirectional exponential moving average (trainable alpha)
    kalman     : forward 1-D Kalman filter, random-walk model (trainable Q, R)
    """

    MODES = {"moving_avg", "gaussian", "learnable", "ema", "kalman"}

    def __init__(self, channels: int, mode: str = "gaussian",
                 kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        assert mode in self.MODES, f"Unknown denoise mode: {mode}"
        self.mode = mode
        self.channels = channels

        if mode == "moving_avg":
            k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = torch.ones(channels, 1, k) / k
            self.register_buffer("kernel", kernel)
            self.pad = k // 2

        elif mode == "gaussian":
            k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            half = k // 2
            x = torch.arange(-half, half + 1, dtype=torch.float32)
            g = torch.exp(-0.5 * (x / sigma) ** 2)
            g = g / g.sum()
            kernel = g.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1).clone()
            self.register_buffer("kernel", kernel)
            self.pad = half

        elif mode == "learnable":
            k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            self.conv = nn.Conv1d(channels, channels, k,
                                  padding=k // 2, groups=channels, bias=False)
            # Initialise with Gaussian
            with torch.no_grad():
                half = k // 2
                x = torch.arange(-half, half + 1, dtype=torch.float32)
                g = torch.exp(-0.5 * (x / sigma) ** 2)
                g = g / g.sum()
                self.conv.weight.data[:] = g.unsqueeze(0).unsqueeze(0)

        elif mode == "ema":
            # Learnable smoothing factor per channel, initialised at 0.3
            self.alpha_logit = nn.Parameter(torch.full((channels,), 0.0))
            # sigmoid(0) = 0.5; we want ~0.3 → logit = ln(0.3/0.7) ≈ -0.847
            nn.init.constant_(self.alpha_logit, math.log(0.3 / 0.7))

        elif mode == "kalman":
            # log-space parameters for positivity
            self.log_Q = nn.Parameter(torch.full((channels,), -2.0))
            self.log_R = nn.Parameter(torch.full((channels,), -1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)"""
        if self.mode in ("moving_avg", "gaussian"):
            return F.conv1d(
                x.transpose(1, 2), self.kernel,
                padding=self.pad, groups=self.channels,
            ).transpose(1, 2)

        if self.mode == "learnable":
            return self.conv(x.transpose(1, 2)).transpose(1, 2)

        if self.mode == "ema":
            return self._ema(x)

        if self.mode == "kalman":
            return self._kalman(x)

        return x

    # -- Bidirectional EMA --------------------------------------------------
    def _ema(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        alpha = torch.sigmoid(self.alpha_logit)  # (C,)

        # Forward
        fwd = [x[:, 0]]
        for t in range(1, T):
            fwd.append(alpha * x[:, t] + (1 - alpha) * fwd[-1])
        fwd = torch.stack(fwd, dim=1)

        # Backward
        bwd = [x[:, -1]]
        for t in range(T - 2, -1, -1):
            bwd.append(alpha * x[:, t] + (1 - alpha) * bwd[-1])
        bwd = torch.stack(list(reversed(bwd)), dim=1)

        return (fwd + bwd) / 2.0

    # -- Forward Kalman filter (random-walk state model) --------------------
    def _kalman(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q = torch.exp(self.log_Q)  # process noise  (C,)
        R = torch.exp(self.log_R)  # observation noise (C,)

        x_hat = [x[:, 0]]                     # initial state = first obs
        P = R.unsqueeze(0).expand(B, -1)       # initial uncertainty = R

        for t in range(1, T):
            # Predict
            P_pred = P + Q                     # (B, C)
            # Update
            K = P_pred / (P_pred + R)          # Kalman gain (B, C)
            innovation = x[:, t] - x_hat[-1]
            x_hat.append(x_hat[-1] + K * innovation)
            P = (1 - K) * P_pred

        return torch.stack(x_hat, dim=1)


class RGBDEncoder(nn.Module):
    """SpatialCNN/ConvNeXtV2/PretrainedCNN per frame -> MomentumMamba temporal -> RMSNorm

    New options:
      encoder="pretrained_cnn": ResNet18 pretrained on ImageNet (best for small datasets)
      freeze="all"/"partial"/"none": freeze strategy for pretrained backbone
      temporal_velocity=True: compute feature-level frame differences (mirrors skeleton velocity)
      temporal_diff_mode="none"/"add"/"gate": feature-level temporal difference with learnable fusion
      use_frame_diff=True: input has extra diff channels (8ch instead of 4ch)
      diff_channels="all"/"rgb_only"/"depth_only": which diff channels are present
    """

    def __init__(
        self,
        d_model=128,
        n_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_momentum=True,
        momentum_mode="real",
        alpha_init=0.6,
        beta_init=0.99,
        encoder="spatial_cnn",
        convnext_model="convnextv2_atto",
        freeze_stages=3,
        freeze="all",
        temporal_velocity=False,
        in_channels=4,
        temporal_diff_mode="none",
        use_frame_diff=False,
        diff_channels="all",
        denoise_mode="none",
        denoise_kernel_size=5,
        denoise_sigma=1.0,
    ):
        super().__init__()
        self.temporal_velocity = temporal_velocity
        self.temporal_diff_mode = temporal_diff_mode

        # Temporal denoising on features (B, N_frames, d_model) after spatial CNN
        if denoise_mode != "none":
            self.denoise = TemporalDenoise(
                channels=d_model, mode=denoise_mode,
                kernel_size=denoise_kernel_size, sigma=denoise_sigma,
            )
        else:
            self.denoise = None

        # Determine actual input channels based on frame diff config
        actual_in_channels = in_channels
        if use_frame_diff:
            if diff_channels == "all":
                actual_in_channels = in_channels + 4   # +ΔRGBD
            elif diff_channels == "rgb_only":
                actual_in_channels = in_channels + 3   # +ΔRGB
            elif diff_channels == "depth_only":
                actual_in_channels = in_channels + 1   # +ΔD

        if encoder == "convnextv2":
            from .backbones import ConvNeXtV2Encoder
            self.spatial = ConvNeXtV2Encoder(
                in_channels=actual_in_channels, feat_dim=d_model,
                model_name=convnext_model, freeze_stages=freeze_stages,
            )
        elif encoder == "pretrained_cnn":
            from .backbones import PretrainedCNN
            self.spatial = PretrainedCNN(
                in_channels=actual_in_channels, d_model=d_model, freeze=freeze,
            )
        else:
            from .backbones import SpatialCNN
            self.spatial = SpatialCNN(in_channels=actual_in_channels, d_model=d_model)

        # Temporal velocity: concat [feat, vel] -> project back to d_model
        if temporal_velocity:
            self.vel_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )

        # Feature-level temporal difference modules (bottleneck design)
        if temporal_diff_mode in ("add", "gate"):
            bottleneck = d_model // 4
            self.motion_net = nn.Sequential(
                nn.Linear(d_model, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, d_model),
            )
        if temporal_diff_mode == "add":
            self.motion_scale = nn.Parameter(torch.tensor(0.1))
        elif temporal_diff_mode == "gate":
            self.motion_gate = nn.Sequential(
                nn.Linear(d_model * 2, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, d_model),
                nn.Sigmoid(),
            )

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode, alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x: (B, N_frames, C, H, W) -> (B, N_frames, d_model)
        C depends on use_frame_diff: 4 (default), 5/7/8 (with diff channels)
        """
        B, N, C, H, W = x.shape
        feat = self.spatial(x.reshape(B * N, C, H, W))
        feat = feat.reshape(B, N, -1)  # (B, N, d_model)

        # Temporal denoising on feature sequence
        if self.denoise is not None:
            feat = self.denoise(feat)

        # Feature-level temporal difference (Approach B)
        if self.temporal_diff_mode in ("add", "gate"):
            feat_diff = torch.zeros_like(feat)
            feat_diff[:, 1:] = feat[:, 1:] - feat[:, :-1]
            # First frame: forward difference (same as second frame's diff)
            if N > 1:
                feat_diff[:, 0] = feat_diff[:, 1]
            motion_feat = self.motion_net(feat_diff)
            if self.temporal_diff_mode == "add":
                feat = feat + self.motion_scale * motion_feat
            else:  # gate
                gate = self.motion_gate(torch.cat([feat, motion_feat], dim=-1))
                feat = feat + gate * motion_feat

        # Feature-level temporal velocity (mirrors skeleton velocity features)
        if self.temporal_velocity:
            vel = torch.zeros_like(feat)
            vel[:, 1:] = feat[:, 1:] - feat[:, :-1]
            feat = self.vel_proj(torch.cat([feat, vel], dim=-1))

        feat = self.drop(feat)
        for blk in self.blocks:
            feat = blk(feat)
        return self.norm(feat)


class IMUEncoder(nn.Module):
    """Conv1D / ConvNeXtV2 (GAF) -> MomentumMamba -> RMSNorm"""

    def __init__(
        self,
        in_channels=6,
        d_model=128,
        n_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_momentum=True,
        momentum_mode="real",
        alpha_init=0.6,
        beta_init=0.99,
        encoder="conv1d",
        convnext_model="convnextv2_atto",
        freeze_stages=3,
        gaf_size=64,
        denoise_mode="none",
        denoise_kernel_size=5,
        denoise_sigma=1.0,
    ):
        super().__init__()
        self.encoder_type = encoder
        self.gaf_size = gaf_size

        # Temporal denoising on raw IMU signal (B, T, 6) before frontend
        if denoise_mode != "none" and encoder != "convnextv2":
            self.denoise = TemporalDenoise(
                channels=in_channels, mode=denoise_mode,
                kernel_size=denoise_kernel_size, sigma=denoise_sigma,
            )
        else:
            self.denoise = None

        if encoder == "convnextv2":
            from .backbones import ConvNeXtV2Encoder
            self.spatial = ConvNeXtV2Encoder(
                in_channels=in_channels, feat_dim=d_model,
                model_name=convnext_model, freeze_stages=freeze_stages,
            )
        else:
            self.frontend = nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode, alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        conv1d mode:      x: (B, T, 6) -> (B, T, d_model)
        convnextv2 mode:  x: (B, 6, H, W) -> (B, 1, d_model)
        """
        if self.encoder_type == "convnextv2":
            feat = self.spatial(x).unsqueeze(1)
            for blk in self.blocks:
                feat = blk(feat)
            return self.norm(feat)
        else:
            # Denoise raw IMU signal before conv frontend
            if self.denoise is not None:
                x = self.denoise(x)
            x = self.frontend(x.transpose(1, 2)).transpose(1, 2)
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)
