"""
MultimodalMMA: RGB-D + IMU multimodal HAR with Momentum Mamba.

Architecture:
  RGB-D (N x 4 x H x W) -> RGBDEncoder (spatial+temporal) --+
                                                              |-> Fusion -> Head -> 27 classes
  IMU   (T x 6)          -> IMUEncoder (Conv1D+Mamba)      --+

Fusion modes (ported from skel_imu best practices):
  - "attention"/"gated": AttentionPool per branch -> DimGatedFusion
  - "cross_mamba": Modality embeddings -> concat -> shared MambaBlock -> AttentionPool
  - "concat": AttentionPool per branch -> MLP

Modality Dropout Schedules:
  - "none": disabled (default, backward-compatible)
  - "fixed": fixed-rate dropout at md_max_p
  - "curriculum": warmup phase (no dropout) then linear ramp to md_max_p
  - "simultaneous": 3 forward passes (full, rgbd-only, imu-only) every iteration
"""

import random

import torch
import torch.nn as nn

from .encoders import RGBDEncoder, IMUEncoder
from .mamba import MomentumMambaBlock
from .mma_skel_imu import AttentionPool, DimGatedFusion, AuxHead


class MultimodalMMA(nn.Module):
    """RGBD+IMU multimodal model with cross_mamba fusion and auxiliary loss."""

    def __init__(
        self,
        num_classes: int = 27,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
        fusion: str = "cross_mamba",
        n_heads: int = 4,
        encoder: str = "default",
        convnext_model: str = "convnextv2_atto",
        freeze_stages: int = 3,
        freeze: str = "all",
        temporal_velocity: bool = False,
        in_channels: int = 4,
        imu_encoder: str = "conv1d",
        gaf_size: int = 64,
        aux_weight: float = 0.0,
        # Modality dropout parameters
        md_schedule: str = "none",
        md_max_p: float = 0.3,
        md_warmup_frac: float = 0.3,
        md_lambda: float = 0.3,
        use_missing_token: bool = True,
        modality_dropout: float = 0.0,  # legacy compat
        # Frame differencing parameters
        use_frame_diff: bool = False,
        diff_channels: str = "all",
        temporal_diff_mode: str = "none",
        # Feature-level modality dropout (MD-Drop)
        md_drop_imu: float = 0.0,
        md_drop_rgbd: float = 0.0,
        # Cross-Modal Alignment Regularization (CMAR)
        cmar_weight: float = 0.0,
        cmar_proj_dim: int = 64,
        cmar_loss_type: str = "mse",  # "mse", "cosine", "barlow"
        md_drop_mode: str = "exclusive",  # "exclusive" or "independent"
        md_drop_curriculum: bool = False,  # ramp up md_drop_imu linearly over training
        md_drop_curriculum_reverse: bool = False,  # start at max, ramp down to 0
        feature_noise_std: float = 0.0,  # Gaussian noise injected into encoded features
        temporal_mask_ratio: float = 0.0,  # Fraction of timesteps to mask in features
        # Temporal denoising filter (applied in encoders at train+eval)
        denoise_mode: str = "none",        # "none","moving_avg","gaussian","learnable","ema","kalman"
        denoise_kernel_size: int = 5,
        denoise_sigma: float = 1.0,
        imu_denoise_mode: str = None,      # per-modality override (None = use denoise_mode)
        rgbd_denoise_mode: str = None,     # per-modality override (None = use denoise_mode)
    ):
        super().__init__()
        self.fusion_mode = fusion
        self.aux_weight = aux_weight
        self._aux_logits = None
        self.d_model = d_model

        # Feature-level noise injection
        self.feature_noise_std = feature_noise_std
        self.temporal_mask_ratio = temporal_mask_ratio

        # Modality dropout config
        self.md_schedule = md_schedule if md_schedule != "none" else (
            "fixed" if modality_dropout > 0 else "none"
        )
        self.md_max_p = md_max_p if md_schedule != "none" else modality_dropout
        self.md_warmup_frac = md_warmup_frac
        self.md_lambda = md_lambda
        self.md_drop_imu = md_drop_imu
        self.md_drop_imu_max = md_drop_imu  # store max for curriculum
        self.md_drop_rgbd = md_drop_rgbd
        self.md_drop_mode = md_drop_mode
        self.md_drop_curriculum = md_drop_curriculum
        self.md_drop_curriculum_reverse = md_drop_curriculum_reverse
        self.cmar_weight = cmar_weight
        self.cmar_loss_type = cmar_loss_type
        self._cmar_loss = None

        # Epoch tracking (set externally by training loop)
        self.current_epoch = 0
        self.total_epochs = 1

        # Learnable missing tokens
        if use_missing_token and self.md_schedule != "none":
            self.missing_token_rgbd = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.missing_token_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.register_buffer("missing_token_rgbd", torch.zeros(1, 1, d_model))
            self.register_buffer("missing_token_imu", torch.zeros(1, 1, d_model))

        mamba_kw = dict(
            d_model=d_model, n_layers=n_layers, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )
        cross_kw = dict(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )

        # Map encoder name
        enc_name = encoder
        if enc_name == "convnextv2":
            enc_name = "convnextv2"
        elif enc_name in ("pretrained", "pretrained_cnn", "resnet18"):
            enc_name = "pretrained_cnn"
        else:
            enc_name = "spatial_cnn"

        # Resolve per-modality denoise modes
        _rgbd_dn = rgbd_denoise_mode if rgbd_denoise_mode is not None else denoise_mode
        _imu_dn = imu_denoise_mode if imu_denoise_mode is not None else denoise_mode

        self.rgbd_enc = RGBDEncoder(
            **mamba_kw,
            encoder=enc_name,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            freeze=freeze,
            temporal_velocity=temporal_velocity,
            in_channels=in_channels,
            temporal_diff_mode=temporal_diff_mode,
            use_frame_diff=use_frame_diff,
            diff_channels=diff_channels,
            denoise_mode=_rgbd_dn,
            denoise_kernel_size=denoise_kernel_size,
            denoise_sigma=denoise_sigma,
        )
        self.imu_enc = IMUEncoder(
            in_channels=6, **mamba_kw,
            encoder=imu_encoder,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            gaf_size=gaf_size,
            denoise_mode=_imu_dn,
            denoise_kernel_size=denoise_kernel_size,
            denoise_sigma=denoise_sigma,
        )

        # --- Fusion ---
        if fusion == "cross_mamba":
            self.mod_embed_rgbd = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.mod_embed_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cross_block = MomentumMambaBlock(**cross_kw)
            self.pool = AttentionPool(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion in ("attention", "gated"):
            self.rgbd_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = DimGatedFusion(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "concat":
            self.rgbd_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = None
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # Auxiliary per-modality heads
        if aux_weight > 0:
            self.aux_rgbd = AuxHead(d_model, num_classes, dropout)
            self.aux_imu = AuxHead(d_model, num_classes, dropout)

        # CMAR projection heads: map each branch to shared alignment space
        if cmar_weight > 0:
            self.cmar_proj_rgbd = nn.Linear(d_model, cmar_proj_dim)
            self.cmar_proj_imu = nn.Linear(d_model, cmar_proj_dim)

    # ------------------------------------------------------------------
    #  Modality dropout helpers
    # ------------------------------------------------------------------

    def get_md_prob(self) -> float:
        """Current modality dropout probability based on schedule and epoch."""
        if self.md_schedule == "none":
            return 0.0
        if self.md_schedule == "fixed" or self.md_schedule == "simultaneous":
            return self.md_max_p
        # curriculum
        warmup_end = int(self.md_warmup_frac * self.total_epochs)
        if self.current_epoch < warmup_end:
            return 0.0
        remaining = self.total_epochs - warmup_end
        if remaining <= 0:
            return self.md_max_p
        progress = (self.current_epoch - warmup_end) / remaining
        return min(self.md_max_p, self.md_max_p * progress)

    def _apply_missing_token(self, feat, token, ref_feat):
        """Replace feat with broadcasted missing token matching ref_feat shape."""
        B, T, _ = ref_feat.shape
        return token.expand(B, T, -1)

    def _fuse_and_classify(self, fv, fi):
        """Run fusion + classification head on encoded features."""
        if self.fusion_mode == "cross_mamba":
            fv2 = fv + self.mod_embed_rgbd
            fi2 = fi + self.mod_embed_imu
            fused = torch.cat([fv2, fi2], dim=1)
            fused = self.cross_block(fused)
            pooled = self.pool(fused)
        elif self.fusion_mode in ("attention", "gated"):
            pv = self.rgbd_pool(fv)
            pi = self.imu_pool(fi)
            pooled = self.fusion_layer(pv, pi)
        else:  # concat
            pv = self.rgbd_pool(fv)
            pi = self.imu_pool(fi)
            pooled = torch.cat([pv, pi], dim=-1)
        return self.head(pooled)

    def forward(self, rgbd, imu):
        """
        rgbd: (B, N, 4, H, W) or None (missing modality)
        imu:  (B, T, 6) or None (missing modality)
        -> logits (B, num_classes)
           OR dict with extra loss terms (simultaneous mode, training only)
        """
        # --- Encode ---
        fv = self.rgbd_enc(rgbd) if rgbd is not None else None
        fi = self.imu_enc(imu) if imu is not None else None

        # Need a reference for shape when building missing tokens
        ref = fv if fv is not None else fi

        # Replace None with missing tokens (inference with missing modality)
        if fv is None:
            fv = self._apply_missing_token(None, self.missing_token_rgbd, ref)
        if fi is None:
            fi = self._apply_missing_token(None, self.missing_token_imu, ref)

        # --- CMAR: Cross-Modal Alignment Regularization ---
        # Must be computed on CLEAN features BEFORE any dropout/masking.
        self._cmar_loss = None
        if self.training and self.cmar_weight > 0:
            fv_proj = self.cmar_proj_rgbd(fv.mean(dim=1))  # (B, proj_dim)
            fi_proj = self.cmar_proj_imu(fi.mean(dim=1))   # (B, proj_dim)
            if self.cmar_loss_type == "cosine":
                cos_sim = nn.functional.cosine_similarity(fv_proj, fi_proj, dim=1)
                self._cmar_loss = torch.mean(1.0 - cos_sim)
            elif self.cmar_loss_type == "barlow":
                fv_n = (fv_proj - fv_proj.mean(0)) / (fv_proj.std(0) + 1e-5)
                fi_n = (fi_proj - fi_proj.mean(0)) / (fi_proj.std(0) + 1e-5)
                cc = (fv_n.T @ fi_n) / fv_n.size(0)
                on_diag = ((1 - cc.diagonal()) ** 2).sum()
                off_diag = (cc.flatten()[1:].view(cc.size(0)-1, cc.size(0)+1)[:, :-1] ** 2).sum()
                self._cmar_loss = on_diag + 0.005 * off_diag
            else:
                self._cmar_loss = torch.mean((fv_proj - fi_proj) ** 2)

        # --- Simultaneous mode (training only) ---
        if self.training and self.md_schedule == "simultaneous":
            return self._forward_simultaneous(fv, fi)

        # --- Feature-level modality dropout (MD-Drop) ---
        if self.training and (self.md_drop_imu > 0 or self.md_drop_rgbd > 0):
            # Curriculum: ramp md_drop_imu from 0 to max over training
            effective_imu_drop = self.md_drop_imu
            if self.md_drop_curriculum and self.total_epochs > 1:
                progress = self.current_epoch / (self.total_epochs - 1)
                effective_imu_drop = self.md_drop_imu_max * progress
            elif self.md_drop_curriculum_reverse and self.total_epochs > 1:
                progress = self.current_epoch / (self.total_epochs - 1)
                effective_imu_drop = self.md_drop_imu_max * (1.0 - progress)
            if self.md_drop_mode == "independent":
                if random.random() < effective_imu_drop:
                    fi = torch.zeros_like(fi)
                if random.random() < self.md_drop_rgbd:
                    fv = torch.zeros_like(fv)
            else:
                r = random.random()
                if r < effective_imu_drop:
                    fi = torch.zeros_like(fi)
                elif r < effective_imu_drop + self.md_drop_rgbd:
                    fv = torch.zeros_like(fv)

        # --- Feature-level noise injection (after CMAR, before fusion) ---
        if self.training and self.feature_noise_std > 0:
            fv = fv + torch.randn_like(fv) * self.feature_noise_std
            fi = fi + torch.randn_like(fi) * self.feature_noise_std

        # --- Temporal feature masking (after CMAR, before fusion) ---
        if self.training and self.temporal_mask_ratio > 0 and random.random() < 0.3:
            B_v, T_v, _ = fv.shape
            n_mask_v = max(1, int(T_v * self.temporal_mask_ratio))
            for b in range(B_v):
                idx = torch.randperm(T_v, device=fv.device)[:n_mask_v]
                fv[b, idx] = 0.0
            B_i, T_i, _ = fi.shape
            n_mask_i = max(1, int(T_i * self.temporal_mask_ratio))
            for b in range(B_i):
                idx = torch.randperm(T_i, device=fi.device)[:n_mask_i]
                fi[b, idx] = 0.0

        # --- Auxiliary outputs (full modality) ---
        self._aux_logits = None
        if self.aux_weight > 0 and self.training:
            self._aux_logits = (self.aux_rgbd(fv), self.aux_imu(fi))

        # --- Random modality dropout (training, fixed/curriculum) ---
        if self.training and self.md_schedule in ("fixed", "curriculum"):
            p = self.get_md_prob()
            if p > 0:
                r = torch.rand(1).item()
                if r < p / 2:
                    # Drop RGBD
                    fv = self._apply_missing_token(fv, self.missing_token_rgbd, fi)
                    # Disable RGBD aux loss
                    if self._aux_logits is not None:
                        self._aux_logits = (None, self._aux_logits[1])
                elif r < p:
                    # Drop IMU
                    fi = self._apply_missing_token(fi, self.missing_token_imu, fv)
                    if self._aux_logits is not None:
                        self._aux_logits = (self._aux_logits[0], None)

        return self._fuse_and_classify(fv, fi)

    def _forward_simultaneous(self, fv, fi):
        """Three-pass forward: full, rgbd-only, imu-only. Returns dict."""
        # Full pass (both modalities present)
        self._aux_logits = None
        if self.aux_weight > 0:
            self._aux_logits = (self.aux_rgbd(fv), self.aux_imu(fi))
        logits_full = self._fuse_and_classify(fv, fi)

        # RGBD-only (IMU replaced with missing token)
        fi_miss = self._apply_missing_token(None, self.missing_token_imu, fv)
        logits_rgbd = self._fuse_and_classify(fv, fi_miss)

        # IMU-only (RGBD replaced with missing token)
        fv_miss = self._apply_missing_token(None, self.missing_token_rgbd, fi)
        logits_imu = self._fuse_and_classify(fv_miss, fi)

        return {
            "logits": logits_full,
            "logits_rgbd_only": logits_rgbd,
            "logits_imu_only": logits_imu,
            "md_lambda": self.md_lambda,
        }
