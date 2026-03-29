import torch
import torch.nn as nn
import torch.nn.functional as F

class UnimodalFeatureEncoder(nn.Module):
    """
    Placeholder for unimodal feature encoders.
    In practice, replace with modality-specific backbones:
    - Visual: ResNet-50 + LSTM or 3D CNN
    - Skeleton/IMU: Co-occurrence features + LSTM + lightweight self-attention

    Args:
        num_modalities: number of input modalities.
        feature_dim: output feature dimension per modality.
        input_dim: int (same for all modalities) or list[int] (per modality).
                   Refers to the feature size *after* temporal mean-pooling.
    """
    def __init__(self, num_modalities: int, feature_dim: int = 128,
                 input_dim: int = 512):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim

        if isinstance(input_dim, (list, tuple)):
            input_dims = list(input_dim)
        else:
            input_dims = [input_dim] * num_modalities

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dims[i], 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim)
            ) for i in range(num_modalities)
        ])
        
    def forward(self, x_list):
        """
        x_list: List of tensors, one per modality [B, seq_len, input_dim_m]
        Returns: unimodal features [B, M, D_u]
        """
        features = []
        for i, x_m in enumerate(x_list):
            # Placeholder: mean pool over sequence (replace with proper spatio-temporal encoding)
            if x_m.dim() > 2:
                x_m = x_m.mean(dim=1)  # simple temporal pooling
            if x_m.dim() > 2:
                x_m = x_m.reshape(x_m.size(0), -1)  # flatten remaining spatial dims
            f_m = self.encoders[i](x_m)
            features.append(f_m)
        
        # Stack to [B, M, D_u]
        return torch.stack(features, dim=1)


class SMFusion(nn.Module):
    """Self Multimodal Fusion (Auxiliary Task)"""
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        # Learnable weights for modality attention (1D conv with kernel=1 or linear)
        self.W_aux = nn.Parameter(torch.randn(feature_dim, 1))
        
    def forward(self, x_u):
        """
        x_u: [B, M, D_u]  unimodal features
        Returns: fused aux features [B, D_u]
        """
        # Compute attention scores: gamma_m = W_aux^T * x_um
        gamma = torch.matmul(x_u, self.W_aux).squeeze(-1)  # [B, M]
        
        # Softmax over modalities
        alpha = F.softmax(gamma, dim=1)  # [B, M]
        
        # Weighted sum
        x_aux = torch.sum(alpha.unsqueeze(-1) * x_u, dim=1)  # [B, D_u]
        return x_aux, alpha  # return alpha for visualization/debug


class GMFusion(nn.Module):
    """Guided Multimodal Fusion (Target Task)"""
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = feature_dim ** 0.5
        
        # Projection matrices
        self.W_Q = nn.Linear(feature_dim, feature_dim)  # for X_aux -> Query
        self.W_K = nn.Linear(feature_dim, feature_dim)  # for X_u -> Key
        self.W_V = nn.Linear(feature_dim, feature_dim)  # for X_u -> Value
        self.W_o = nn.Linear(feature_dim, feature_dim)  # output projection
        
    def forward(self, x_u, x_aux):
        """
        x_u: [B, M, D_u]
        x_aux: [B, D_u]
        Returns: guided fused features [B, D_u]
        """
        # Project to Q, K, V
        Q = self.W_Q(x_aux).unsqueeze(1)          # [B, 1, D_u]
        K = self.W_K(x_u)                         # [B, M, D_u]
        V = self.W_V(x_u)                         # [B, M, D_u]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, 1, M]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Attend to values
        x_c_prime = torch.matmul(attn_weights, V).squeeze(1)  # [B, D_u]
        
        # Output projection
        x_c = self.W_o(x_c_prime)
        return x_c, attn_weights


class MuMu(nn.Module):
    """MuMu: Cooperative Multitask Multimodal Fusion

    Args:
        num_modalities: number of input modalities.
        feature_dim: hidden / fusion feature dimension.
        num_activity_groups: classes for auxiliary head.
        num_activities: classes for target head.
        input_dim: int or list[int] — per-modality input dim after temporal
                   mean-pooling.  Passed to UnimodalFeatureEncoder.
    """
    def __init__(self, 
                 num_modalities: int, 
                 feature_dim: int = 128,
                 num_activity_groups: int = 10,   # depends on your dataset
                 num_activities: int = 20,        # depends on your dataset
                 input_dim: int = 512):            # per-modality input feature dim
        super().__init__()
        
        self.feature_dim = feature_dim
        
        self.ufe = UnimodalFeatureEncoder(num_modalities, feature_dim, input_dim)
        self.sm_fusion = SMFusion(feature_dim)
        self.gm_fusion = GMFusion(feature_dim)
        
        # Auxiliary head (activity groups)
        self.aux_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_activity_groups)
        )
        
        # Target head (fine-grained activities)
        self.target_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # after concat
            nn.ReLU(),
            nn.Linear(feature_dim, num_activities)
        )
        
    def forward(self, x_list):
        """
        x_list: List of modality inputs
        Returns: 
            y_aux (activity group logits),
            y_target (activity logits),
            attention weights (for analysis)
        """
        # Step 1: Unimodal features
        x_u = self.ufe(x_list)                    # [B, M, D_u]
        
        # Step 2: Auxiliary Task - SM-Fusion
        x_aux, alpha = self.sm_fusion(x_u)        # [B, D_u]
        y_aux = self.aux_head(x_aux)
        
        # Step 3: Target Task - GM-Fusion guided by x_aux
        x_c, attn_weights = self.gm_fusion(x_u, x_aux)  # [B, D_u]
        
        # Concatenate guided fusion + aux features
        x_f = torch.cat([x_c, x_aux], dim=1)      # [B, 2*D_u]
        y_target = self.target_head(x_f)
        
        return y_aux, y_target, alpha, attn_weights


# ============== Training Example ==============
class MuMuLoss(nn.Module):
    def __init__(self, beta_aux: float = 0.5):
        super().__init__()
        self.beta_aux = beta_aux
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, y_aux_pred, y_aux_true, y_target_pred, y_target_true):
        loss_aux = self.ce(y_aux_pred, y_aux_true)
        loss_target = self.ce(y_target_pred, y_target_true)
        return loss_target + self.beta_aux * loss_aux


# Usage example
if __name__ == "__main__":
    batch_size = 8
    num_mod = 3          # e.g., RGB, Skeleton, IMU
    feature_dim = 128
    
    model = MuMu(num_modalities=num_mod, 
                 feature_dim=feature_dim,
                 num_activity_groups=8,
                 num_activities=30)
    
    # Dummy inputs: list of [B, seq, feat] per modality
    dummy_inputs = [
        torch.randn(batch_size, 16, 512) for _ in range(num_mod)
    ]
    
    y_aux, y_target, alpha, attn = model(dummy_inputs)
    
    print("Aux logits shape:", y_aux.shape)
    print("Target logits shape:", y_target.shape)
    print("Modality attention (SM):", alpha.shape)
    print("Guided attention (GM):", attn.shape)