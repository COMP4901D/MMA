"""
MuMu: Cooperative Multitask Learning-Based Guided Multimodal Fusion
====================================================================
Reference: Islam & Iqbal, AAAI 2022.
           https://doi.org/10.1609/aaai.v36i1.19988

Architecture (Figure 2 in paper):
    1. Unimodal Feature Encoders (UFE): Bi-LSTM + temporal self-attention
    2. Self-Multimodal Fusion (SM-Fusion): attention-weighted modality fusion → auxiliary task
    3. Guided Multimodal Fusion (GM-Fusion): cross-attention guided by auxiliary features → target task
    4. Cooperative loss: L = L_target + β · L_aux   (defined in train/mumu_train.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Temporal Self-Attention (Section 3.1, Eq. 1-2) ─────────────

class TemporalSelfAttention(nn.Module):
    """Lightweight self-attention over temporal LSTM hidden states.

    α_t = softmax(w^T tanh(W_h h_t + b))
    c   = Σ α_t · h_t
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor):
        """
        h: [B, T, H]
        Returns: context [B, H], temporal attention weights [B, T]
        """
        scores = self.attn(h).squeeze(-1)          # [B, T]
        alpha = F.softmax(scores, dim=1)            # [B, T]
        context = torch.sum(alpha.unsqueeze(-1) * h, dim=1)  # [B, H]
        return context, alpha


# ── Unimodal Feature Encoder (Section 3.1) ─────────────────────

class UnimodalFeatureEncoder(nn.Module):
    """Per-modality encoder: Bi-LSTM → temporal self-attention → projection.

    Args:
        num_modalities: number of input modalities.
        feature_dim: output dim per modality.
        input_dim: int or list[int], per-modality input feature size.
        hidden_dim: LSTM hidden size (per direction).
        num_lstm_layers: number of LSTM layers.
        dropout: dropout probability.
    """

    def __init__(self, num_modalities: int, feature_dim: int = 128,
                 input_dim: int = 6, hidden_dim: int = 128,
                 num_lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_modalities = num_modalities

        if isinstance(input_dim, (list, tuple)):
            input_dims = list(input_dim)
        else:
            input_dims = [input_dim] * num_modalities

        self.lstms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.projs = nn.ModuleList()

        for i in range(num_modalities):
            lstm_out = hidden_dim * 2  # bidirectional
            self.lstms.append(nn.LSTM(
                input_size=input_dims[i],
                hidden_size=hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_lstm_layers > 1 else 0.0,
            ))
            self.attns.append(TemporalSelfAttention(lstm_out))
            self.projs.append(nn.Sequential(
                nn.LayerNorm(lstm_out),
                nn.Linear(lstm_out, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x_list):
        """
        x_list: list of [B, T_m, D_m] per modality.
        Returns: [B, M, feature_dim]
        """
        features = []
        for i, x_m in enumerate(x_list):
            if x_m.dim() == 2:
                x_m = x_m.unsqueeze(1)         # [B, D] → [B, 1, D]
            h, _ = self.lstms[i](x_m)          # [B, T, 2H]
            c, _ = self.attns[i](h)            # [B, 2H]
            f_m = self.projs[i](c)             # [B, feature_dim]
            features.append(f_m)
        return torch.stack(features, dim=1)    # [B, M, feature_dim]


# ── SM-Fusion (Section 3.2, Eq. 3-5) ───────────────────────────

class SMFusion(nn.Module):
    """Self-Multimodal Fusion for auxiliary (activity-group) task.

    γ_m = w^T f_m       modality score
    α   = softmax(γ)    modality attention
    x_aux = Σ α_m f_m   fused representation
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x_u):
        """
        x_u: [B, M, D]
        Returns: x_aux [B, D], alpha [B, M]
        """
        gamma = self.gate(x_u).squeeze(-1)         # [B, M]
        alpha = F.softmax(gamma, dim=1)             # [B, M]
        x_aux = torch.sum(alpha.unsqueeze(-1) * x_u, dim=1)  # [B, D]
        return x_aux, alpha


# ── GM-Fusion (Section 3.3, Eq. 6-9) ───────────────────────────

class GMFusion(nn.Module):
    """Guided Multimodal Fusion for target (fine-grained activity) task.

    Q = W_Q x_aux          auxiliary features guide the query
    K = W_K X_u,  V = W_V X_u
    x_c = W_o · softmax(Q K^T / √d) · V
    """

    def __init__(self, feature_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scale = feature_dim ** 0.5
        self.W_Q = nn.Linear(feature_dim, feature_dim)
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_V = nn.Linear(feature_dim, feature_dim)
        self.W_o = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_u, x_aux):
        """
        x_u: [B, M, D],  x_aux: [B, D]
        Returns: x_c [B, D], attn_weights [B, 1, M]
        """
        Q = self.W_Q(x_aux).unsqueeze(1)                         # [B, 1, D]
        K = self.W_K(x_u)                                         # [B, M, D]
        V = self.W_V(x_u)                                         # [B, M, D]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, 1, M]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        x_c = torch.matmul(attn_weights, V).squeeze(1)           # [B, D]
        x_c = self.W_o(x_c)
        return x_c, attn_weights


# ── MuMu Model ─────────────────────────────────────────────────

class MuMu(nn.Module):
    """MuMu: Cooperative Multitask Learning-Based Guided Multimodal Fusion.

    Architecture:
        x_list → UFE → x_u ──→ SM-Fusion → x_aux → AuxHead → y_aux
                            └─→ GM-Fusion(x_u, x_aux) → x_c
                                    [x_c; x_aux] → TargetHead → y_target

    Args:
        num_modalities: number of input modalities.
        feature_dim: fusion feature dimension.
        num_activity_groups: classes for auxiliary head.
        num_activities: classes for target head.
        input_dim: int or list[int] — per-modality raw feature dim.
        hidden_dim: LSTM hidden size (per direction).
        num_lstm_layers: stacked LSTM layers.
        dropout: dropout rate.
    """

    def __init__(self,
                 num_modalities: int = 1,
                 feature_dim: int = 128,
                 num_activity_groups: int = 5,
                 num_activities: int = 27,
                 input_dim: int = 6,
                 hidden_dim: int = 128,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim

        self.ufe = UnimodalFeatureEncoder(
            num_modalities, feature_dim, input_dim,
            hidden_dim, num_lstm_layers, dropout,
        )
        self.sm_fusion = SMFusion(feature_dim)
        self.ln_aux = nn.LayerNorm(feature_dim)

        self.gm_fusion = GMFusion(feature_dim, dropout=dropout * 0.5)
        self.ln_gm = nn.LayerNorm(feature_dim)

        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_activity_groups),
        )
        self.target_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_activities),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_list):
        """
        x_list: list of [B, T_m, D_m] per modality.
        Returns: (y_aux, y_target, alpha, attn_weights)
        """
        x_u = self.ufe(x_list)                         # [B, M, D]

        x_aux, alpha = self.sm_fusion(x_u)              # [B, D], [B, M]
        x_aux = self.ln_aux(x_aux)
        y_aux = self.aux_head(x_aux)

        x_c, attn_weights = self.gm_fusion(x_u, x_aux) # [B, D], [B, 1, M]
        x_c = self.ln_gm(x_c)

        x_f = torch.cat([x_c, x_aux], dim=1)           # [B, 2D]
        y_target = self.target_head(x_f)

        return y_aux, y_target, alpha, attn_weights


# ── Usage Example ───────────────────────────────────────────────
if __name__ == "__main__":
    B, M = 8, 1
    model = MuMu(
        num_modalities=M, feature_dim=128,
        num_activity_groups=5, num_activities=27, input_dim=6,
    )

    dummy = [torch.randn(B, 64, 6) for _ in range(M)]
    y_aux, y_target, alpha, attn = model(dummy)

    print(f"Aux logits:    {y_aux.shape}")      # [8, 5]
    print(f"Target logits: {y_target.shape}")    # [8, 27]
    print(f"SM attention:  {alpha.shape}")       # [8, 1]
    print(f"GM attention:  {attn.shape}")        # [8, 1, 1]

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")