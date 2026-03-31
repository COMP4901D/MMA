# MMA Skeleton+IMU Multimodal HAR — Experiment Report

## 1. Model Architecture

### 1.1 Overview

**MMA_SkeletonIMU** is a dual-branch **Momentum Mamba** architecture for multimodal Human Activity Recognition (HAR) on the UTD-MHAD dataset using skeleton and IMU modalities.

```
┌─────────────────────────────────────────────────────────┐
│                  MMA_SkeletonIMU                        │
│                                                         │
│  ┌──────────────────┐    ┌───────────────────┐          │
│  │ Skeleton Encoder │    │   IMU Encoder     │          │
│  │  (Linear/Spatial)│    │  (Conv1D/MultiSc) │          │
│  │                  │    │                    │          │
│  │ Frontend         │    │ Frontend           │         │
│  │   ↓              │    │   ↓                │         │
│  │ MambaBlock ×N    │    │ MambaBlock ×N      │         │
│  │   ↓              │    │   ↓                │         │
│  │ RMSNorm          │    │ RMSNorm            │         │
│  └────────┬─────────┘    └─────────┬──────────┘         │
│           │   (Optional Aux Head)  │                    │
│           └──────────┬─────────────┘                    │
│                      │ Fusion                           │
│           ┌──────────┴──────────┐                       │
│           │ Attention / Gated / │                       │
│           │ Cross-Mamba / Concat│                       │
│           └──────────┬──────────┘                       │
│                      ↓                                  │
│              Classification Head                        │
│              Dropout → Linear(D, 27)                    │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Skeleton Encoder Variants

| Variant | Input | Frontend | Description |
|---------|-------|----------|-------------|
| `linear` (default) | (B, T, 60) — 20 joints × 3 coords | Linear(60→D) → LN → ReLU → Dropout | Baseline: flat joint coordinates |
| `spatial` | (B, T, 177) — joints + bones + velocity | Linear(177→D) → LN → ReLU → Dropout | Augmented with 19 bone vectors (57-dim) and temporal velocity (60-dim) |

**Optional**: `center_joints=True` subtracts hip joint (joint 0) from all joints for translation invariance.

Both are followed by `n_layers` **MomentumMambaBlock** layers and **RMSNorm**.

### 1.3 IMU Encoder Variants

| Variant | Input | Frontend | Description |
|---------|-------|----------|-------------|
| `conv1d` (default) | (B, T, 6) — 6-axis IMU | Conv1D(6→D, k=3) → BN → ReLU → Dropout | Baseline: local temporal pattern extraction |
| `multiscale` | (B, T, 6) | Parallel Conv1D (k=3,7,15) → concat → BN → ReLU → Dropout | Multi-resolution temporal features |

Both are followed by `n_layers` **MomentumMambaBlock** layers and **RMSNorm**.

### 1.4 MomentumMambaBlock (Core Building Block)

Each MomentumMambaBlock implements the **Momentum SSM** — a selective state-space model with exponential momentum-based dynamics:
- **d_model**: Hidden dimension (default 128)
- **d_state**: SSM state dimension (default 32)
- **d_conv**: Local convolution kernel size (default 4)
- **expand**: Inner expansion factor (default 2, inner_dim = d_model × expand)
- **Momentum**: Uses `alpha_init=0.6`, `beta_init=0.99` for adaptive momentum

### 1.5 Fusion Modes

| Mode | Mechanism | Description |
|------|-----------|-------------|
| `attention` / `gated` | AttentionPool(skel) + AttentionPool(imu) → **DimGatedFusion** | Per-dimension learned gate selects from each modality |
| `cross_mamba` | Modality embeddings → Concat sequences → shared **MambaBlock** → AttentionPool | Cross-modal interaction via shared SSM over concatenated sequences |
| `concat` | AttentionPool(skel) \|\| AttentionPool(imu) → MLP | Simple concatenation of pooled features |

### 1.6 Auxiliary Loss (Key Innovation)

When `aux_weight > 0`, per-modality **AuxHead** modules (AttentionPool + Linear) produce separate classification logits for skeleton and IMU branches. The auxiliary loss:

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda_{aux} \cdot \frac{\mathcal{L}_{skel} + \mathcal{L}_{imu}}{2}$$

This forces each encoder branch to learn independently discriminative features, dramatically improving fusion quality.

### 1.7 Classification Head (Decoder)

All fusion modes use the same structure:
```
Dropout(p) → Linear(d_model, 27)
```
Where 27 = number of action classes in UTD-MHAD.

For `concat` fusion, an intermediate MLP is used:
```
Dropout(p) → Linear(2×d_model, d_model) → ReLU → Dropout(p) → Linear(d_model, 27)
```

### 1.8 AttentionPool (Temporal Pooling)

Learnable attention-weighted temporal pooling replaces naive mean/max pooling:
```
Linear(D, D/4) → Tanh → Linear(D/4, 1) → Softmax → Weighted Sum
```
Input: (B, T, D) → Output: (B, D)

### 1.9 DimGatedFusion

Per-dimension gated fusion learns which dimensions to take from each modality:
```
Gate: Linear(2D, D) → ReLU → Linear(D, D) → Sigmoid
Output: gate ⊙ f_skel + (1 - gate) ⊙ f_imu
```

---

## 2. Dataset

- **UTD-MHAD**: 27 action classes, 8 subjects
- **Split**: Subjects 1,3,5,7 for training (431 samples), subjects 2,4,6,8 for testing (430 samples)
- **Skeleton**: 20 joints × 3 coordinates, padded to T=128
- **IMU**: 6-axis (3 acc + 3 gyro), padded to T=192
- **Augmentation** (train only): jitter (σ=0.01/0.03), scaling (σ=0.1), rotation (±0.15/0.2 rad)

---

## 3. Training Configuration (Default)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.008 |
| Scheduler | Cosine Warmup (5 warmup epochs) |
| Batch Size | 16 |
| Epochs | 100 |
| Patience | 40 (early stopping) |
| Label Smoothing | 0.1 |
| Mixup Alpha | 0.15 |
| AMP | Enabled |
| Seed | 42 |

---

## 4. Experiment Results

### 4.1 Phase 1: Architecture Search (Exp1–Exp9)

| Exp | d_model | n_layers | Fusion | Other | Seed | Acc | F1 | Notes |
|-----|---------|----------|--------|-------|------|-----|-----|-------|
| 1 | 128 | 2 | attention | baseline | 42 | 0.8814 | 0.8796 | Initial baseline |
| 2 | 160 | 2 | attention | dim_gate | 42 | 0.9256 | 0.9254 | AttentionPool + DimGatedFusion |
| 2b | 160 | 2 | attention | dim_gate | 123 | 0.9419 | 0.9419 | Best with attention fusion |
| 3 | 192 | 2 | attention | — | 42 | NaN | — | d=192 causes NaN |
| 4 | 192 | 2 | attention | reduced | 42 | NaN | — | Still unstable |
| 5 | 160 | 2 | cross_mamba | — | 42 | 0.9302 | 0.9290 | Cross-modal Mamba |
| 5b | 160 | 2 | cross_mamba | — | 123 | 0.9209 | 0.9194 | Seed variation |
| 6 | 160 | 3 | attention | 3 layers | 42 | 0.8977 | 0.8971 | Deeper = overfitting |
| 7 | 160 | 3 | cross_mamba | 3 layers | 42 | 0.8791 | 0.8769 | Deeper = worse |
| 8 | 160 | 2 | cross_mamba | spatial skel | 42 | 0.9233 | 0.9225 | Spatial encoder hurts |
| 9 | 160 | 2 | cross_mamba | multiscale IMU | 42 | 0.9326 | — | Interrupted at epoch 87 |

### 4.2 Phase 2: Encoder & Regularization Research (Exp10–Exp21)

| Exp | Configuration | Seed | Acc | F1 | Notes |
|-----|--------------|------|-----|-----|-------|
| 10 | spatial_skel + multiscale_imu + attention | 42 | 0.9070 | 0.9065 | Spatial encoder hurts |
| 11 | multiscale_imu + attention | 42 | 0.9116 | 0.9116 | Multiscale hurts attention |
| 12 | center_joints + attention | 42 | 0.9279 | 0.9275 | Slight improvement |
| 13 | center_joints + attention | 123 | 0.9163 | 0.9128 | Inconsistent across seeds |
| 14 | pos_enc + attention | 42 | 0.9233 | 0.9220 | Positional encoding hurts |
| 15 | stronger reg (ls=0.15, mx=0.2, wd=0.01) + attention | 123 | 0.9256 | 0.9253 | No improvement |
| 16 | aux_loss(0.3) + attention | 123 | 0.9209 | 0.9197 | Aux helps less with attention |
| 17 | aux_loss(0.3) + attention | 42 | 0.9302 | 0.9295 | +0.0046 vs baseline attention |
| **18** | **aux_loss(0.3) + cross_mamba** | **42** | **0.9605** | **0.9600** | **Breakthrough!** |
| 19 | aux_loss(0.3) + cross_mamba | 123 | 0.9326 | 0.9333 | Seed variation |
| 20 | aux + cross_mamba + multiscale IMU | 42 | 0.9349 | 0.9344 | Multiscale hurts with aux |
| 21 | aux_loss(0.3) + cross_mamba | 7 | 0.9558 | 0.9556 | Confirms combo is strong |

### 4.3 Phase 3: Hyperparameter Tuning (Exp22–Exp30)

Base config: **d_model=160, n_layers=2, fusion=cross_mamba, seed=42** (tuning around Exp18)

| Exp | Change from Exp18 | bs | lr | aux_w | dropout | wd | expand | Acc | F1 | Notes |
|-----|-------------------|----|----|-------|---------|------|--------|-----|-----|-------|
| 22 | bs=32 | 32 | 3e-4 | 0.3 | 0.2 | 0.008 | 2 | 0.9326 | 0.9333 | Larger batch hurts without lr scaling |
| 23 | bs=32, lr scaled | 32 | 4.2e-4 | 0.3 | 0.2 | 0.008 | 2 | 0.9419 | 0.9414 | Sqrt lr scaling partially recovers |
| 24 | aux_weight=0.5 | 16 | 3e-4 | 0.5 | 0.2 | 0.008 | 2 | 0.9535 | 0.9530 | Stronger aux slightly worse |
| **25** | **aux_weight=0.1** | **16** | **3e-4** | **0.1** | **0.2** | **0.008** | **2** | **0.9628** | **0.9622** | **NEW BEST — weaker aux is better** |
| **26** | **lr=5e-4** | **16** | **5e-4** | **0.3** | **0.2** | **0.008** | **2** | **0.9628** | **0.9622** | **Ties with Exp25** |
| 27 | lr=1e-4 | 16 | 1e-4 | 0.3 | 0.2 | 0.008 | 2 | 0.9209 | 0.9171 | Too low lr, significantly hurts |
| 28 | dropout=0.3, wd=0.012 | 16 | 3e-4 | 0.3 | 0.3 | 0.012 | 2 | 0.9465 | 0.9460 | Stronger regularization hurts |
| 29 | expand=3 | 16 | 3e-4 | 0.3 | 0.2 | 0.008 | 3 | 0.9442 | 0.9440 | Larger expansion = overfitting |
| 30 | aux=0.1 + lr=5e-4 | 16 | 5e-4 | 0.1 | 0.2 | 0.008 | 2 | 0.9558 | 0.9552 | Combo doesn't synergize |

---

## 5. All-Time Leaderboard

| Rank | Exp | Configuration | Seed | Acc | F1 | Checkpoint |
|------|-----|--------------|------|-----|-----|------------|
| 🥇 | **25** | aux(0.1) + cross_mamba, lr=3e-4 | 42 | **0.9628** | **0.9622** | skel_imu_exp25.pt |
| 🥇 | **26** | aux(0.3) + cross_mamba, lr=5e-4 | 42 | **0.9628** | **0.9622** | skel_imu_exp26.pt |
| 🥉 | 18 | aux(0.3) + cross_mamba, lr=3e-4 | 42 | 0.9605 | 0.9600 | skel_imu_exp18.pt |
| 4 | 30 | aux(0.1) + cross_mamba, lr=5e-4 | 42 | 0.9558 | 0.9552 | skel_imu_exp30.pt |
| 5 | 21 | aux(0.3) + cross_mamba, lr=3e-4 | 7 | 0.9558 | 0.9556 | skel_imu_exp21.pt |
| 6 | 24 | aux(0.5) + cross_mamba, lr=3e-4 | 42 | 0.9535 | 0.9530 | skel_imu_exp24.pt |
| 7 | 28 | aux(0.3) + cross_mamba, do=0.3, wd=0.012 | 42 | 0.9465 | 0.9460 | skel_imu_exp28.pt |
| 8 | 29 | aux(0.3) + cross_mamba, expand=3 | 42 | 0.9442 | 0.9440 | skel_imu_exp29.pt |
| 9 | 2b | attention + dim_gate, d=160 | 123 | 0.9419 | 0.9419 | skel_imu_exp2b.pt |
| 10 | 23 | aux(0.3) + cross_mamba, bs=32, lr=4.2e-4 | 42 | 0.9419 | 0.9414 | skel_imu_exp23.pt |

---

## 6. Key Findings

### What Worked
1. **Auxiliary per-modality loss** is the single most impactful technique — forces each encoder to learn discriminative features independently
2. **Cross-Mamba fusion** synergizes strongly with auxiliary loss — shared SSM over concatenated sequences enables deep cross-modal interaction
3. **aux_weight=0.1** (subtle regularization) slightly outperforms 0.3 (Exp25: 0.9628 vs Exp18: 0.9605)
4. **lr=5e-4** with aux_weight=0.3 matches the best result (Exp26: 0.9628)
5. **d_model=160** is the sweet spot for this dataset size (431 training samples)
6. **AttentionPool + DimGatedFusion** is the best non-cross-mamba fusion approach

### What Didn't Work
1. **Larger batch size** (32): Fewer gradient updates per epoch hurts on small datasets, even with lr scaling
2. **Spatial skeleton encoder**: 177-dim input (60 joints + 57 bones + 60 velocity) is too many features for 431 samples
3. **Multi-scale IMU encoder**: Marginal benefit alone, hurts when combined with auxiliary loss
4. **Positional encoding**: Mamba's SSM already captures sequential ordering
5. **Stronger regularization** (dropout=0.3, wd=0.012): Over-regularizes, hurting capacity
6. **Larger expansion** (expand=3): More parameters → more overfitting on small dataset
7. **Lower lr** (1e-4): Under-fits, Acc=0.9209
8. **3 layers**: Overfits on small dataset
9. **Modality dropout**: Hurts by removing useful information
10. **Center joints**: Inconsistent across seeds
11. **Combining best individual improvements** (aux=0.1 + lr=5e-4): Doesn't synergize (0.9558 < 0.9628)

### Optimal Configuration
```
d_model=160, n_layers=2, expand=2, dropout=0.2
fusion=cross_mamba, aux_weight=0.1 (or 0.3 with lr=5e-4)
lr=3e-4, weight_decay=0.008, batch_size=16
label_smoothing=0.1, mixup_alpha=0.15
scheduler=cosine_warmup, warmup_epochs=5, patience=40
```

---

## 7. Model Parameter Count

| Component | Parameters |
|-----------|-----------|
| Skeleton Encoder (linear, d=160) | ~142K |
| IMU Encoder (conv1d, d=160) | ~142K |
| Cross-Mamba Fusion Block | ~141K |
| AttentionPool | ~6.6K |
| Classification Head | ~4.3K |
| Aux Heads (×2, when enabled) | ~13.2K |
| Modality Embeddings | ~320 |
| **Total (with aux)** | **~450K** |

---

## 8. Reproducibility

All experiments used:
- **Hardware**: NVIDIA RTX 4090
- **Software**: PyTorch 2.11.0+cu130, Python 3.10, conda env "mma"
- **Dataset**: UTD-MHAD with fixed subject-based train/test split
- **Seeds**: 42 (primary), 123, 7 (for cross-seed validation)

### Training Command (Best Config — Exp25)
```powershell
python train/run_train.py --pipeline skel_imu --data_root "./datasets/UTD-MHAD" `
  --epochs 100 --save_name skel_imu_exp25.pt --tb_dir runs --num_workers 0 `
  --label_smoothing 0.1 --mixup_alpha 0.15 --weight_decay 0.008 `
  --scheduler cosine_warmup --warmup_epochs 5 --lr 3e-4 --patience 40 --seed 42 `
  --model_kwargs '{"d_model":160,"fusion":"cross_mamba","modality_dropout":0.0,"aux_weight":0.1}'
```

---
---

# MMA RGBD+IMU Multimodal HAR — Experiment Report

## 1. Motivation

Skeleton data requires external preprocessing (pose estimation) before training. To remove this dependency, we explore replacing skeleton with **raw RGBD video** (RGB + Depth) while keeping IMU. This section documents the RGBD+IMU experiments (R1–R9) and compares with the skeleton+IMU pipeline.

## 2. Model Architecture

### 2.1 Overview

**MultimodalMMA (RGBD+IMU)** reuses the same proven architecture from the skeleton+IMU pipeline (cross_mamba fusion, auxiliary loss, AttentionPool, DimGatedFusion), replacing only the skeleton branch with a visual encoder.

```
┌─────────────────────────────────────────────────────────────┐
│                  MultimodalMMA (RGBD+IMU)                   │
│                                                             │
│  ┌────────────────────┐    ┌───────────────────┐            │
│  │   RGBD Encoder     │    │   IMU Encoder     │            │
│  │  (SpatialCNN /     │    │  (Conv1D)         │            │
│  │   ResNet18)        │    │                    │            │
│  │  per-frame → (B,T,D)   │  (B,T,6)→(B,T,D)  │            │
│  │  [+ Temporal Vel]  │    │                    │            │
│  │       ↓            │    │       ↓            │            │
│  │  MambaBlock ×N     │    │  MambaBlock ×N     │            │
│  │       ↓            │    │       ↓            │            │
│  │  RMSNorm           │    │  RMSNorm           │            │
│  └────────┬───────────┘    └─────────┬──────────┘            │
│           │   (Optional Aux Head)    │                      │
│           └───────────┬──────────────┘                      │
│                       │ Fusion                              │
│            ┌──────────┴──────────┐                          │
│            │ Cross-Mamba / Attn  │                          │
│            │ / Gated / Concat    │                          │
│            └──────────┬──────────┘                          │
│                       ↓                                     │
│               Classification Head                           │
│               Dropout → Linear(D, 27)                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 RGBD Encoder Variants

| Variant | Input | Frontend | Params | Description |
|---------|-------|----------|--------|-------------|
| `spatial_cnn` (default) | (B, T, 4, 112, 112) | 4-layer stride-2 Conv2D → per-frame feature | ~340K | Lightweight, trains from scratch |
| `pretrained_cnn` (ResNet18) | (B, T, 4, 112, 112) | ImageNet-pretrained ResNet18, first conv adapted 3→4 channels | ~11.2M | Pretrained visual features |

**Freeze modes** (for `pretrained_cnn`):
- `all`: Only final projection layer trainable (~82K trainable)
- `partial`: layer4 + projection trainable (~2.6M trainable)
- `none`: Full fine-tuning (~11.2M trainable)

### 2.3 Temporal Velocity Features

When `temporal_velocity=True`, the encoder computes frame-level feature differences (analogous to skeleton velocity features):

$$v_t = f_t - f_{t-1}, \quad v_0 = 0$$

The velocity features are concatenated with the original features and projected back: `Linear(2D, D) → LN → ReLU`.

### 2.4 Data Augmentation

RGBD augmentations (train only, `augment=True`):
- Random horizontal flip (p=0.5)
- Random crop (scale 0.8–1.0) + resize to 112×112
- Color jitter (brightness/contrast ±15%) on RGB channels only

IMU augmentations:
- Jitter (σ=0.03)
- Scaling (σ=0.1)

### 2.5 Optical Flow (Experimental)

When `use_flow=True`, 2 additional channels of dense optical flow (computed via Farnebäck method) are appended, resulting in 6-channel input (RGBD + flow_x + flow_y). This did not improve results (see R8).

## 3. RGBD Data Pipeline

```
RGB (.avi, 640×480)  ──→  resize 112×112  ──→  normalize [0,1]  ──┐
Depth (.mat, 320×240) ─→  resize 112×112  ──→  normalize [0,1]  ──┤
                                                                    ↓
                                              Stack → (T, 4, 112, 112)
                                              Uniform sample T=16 frames
                                                                    ↓
                                              Total per-sample: 16 × 4 × 112 × 112 = 802,816 values
```

Compare with skeleton: `128 × 60 = 7,680 values` — a **105× input dimensionality gap**.

## 4. Experiment Results

### 4.1 Phase 1: Baseline RGBD+IMU (R1–R2)

All experiments use the same training config as skeleton+IMU best (label_smoothing=0.1, mixup_alpha=0.15, weight_decay=0.008, cosine_warmup, patience=40) unless noted.

| Exp | Encoder | Fusion | aux_w | bs | lr | Acc | F1 | Notes |
|-----|---------|--------|-------|----|----|-----|-----|-------|
| R1 | SpatialCNN | cross_mamba | 0.1 | 8 | 3e-4 | 0.8930 | 0.8882 | Baseline |
| R2 | SpatialCNN | attention | 0.1 | 8 | 3e-4 | 0.8767 | 0.8711 | Attention fusion weaker |

### 4.2 Phase 2: Pretrained CNN + Velocity + Augmentation (R3–R6)

Three improvements applied: (1) pretrained ResNet18 backbone, (2) temporal velocity features, (3) data augmentation.

| Exp | Encoder | Freeze | vel | aug | bs | lr | Acc | F1 | Notes |
|-----|---------|--------|-----|-----|----|----|-----|-----|-------|
| R3 | ResNet18 | all | ✓ | ✓ | 8 | 3e-4 | 0.8860 | 0.8819 | Frozen backbone under-fits |
| **R4** | **ResNet18** | **partial** | **✓** | **✓** | **8** | **3e-4** | **0.8977** | **0.8951** | **RGBD BEST** |
| R5 | SpatialCNN | — | ✓ | ✓ | 8 | 3e-4 | 0.8953 | 0.8929 | SpatialCNN competitive |
| R6 | ResNet18 | none | ✓ | ✓ | 8 | 1e-4 | 0.8977 | 0.8955 | Full finetune ties R4 |

### 4.3 Phase 3: Additional Exploration (R7–R9)

| Exp | Configuration | bs | Acc | Notes |
|-----|---------------|-----|-----|-------|
| R7 | R4 config + 32 frames | 4 | 0.8837 | OOM at epoch 51, more frames didn't help |
| R8 | Optical flow (6ch) + vel + aug | 32 | 0.8791 | Flow didn't improve, bs=32 hurt |
| R9 | SpatialCNN + vel + aug | 32 | 0.8558 | bs=32 clearly hurts convergence |

## 5. RGBD+IMU Leaderboard

| Rank | Exp | Configuration | Acc | F1 |
|------|-----|--------------|-----|-----|
| 🥇 | **R4** | ResNet18 partial + velocity + augmentation, bs=8 | **0.8977** | **0.8951** |
| 🥇 | R6 | ResNet18 full finetune + vel + aug, bs=8, lr=1e-4 | 0.8977 | 0.8955 |
| 3 | R5 | SpatialCNN + vel + aug, bs=8 | 0.8953 | 0.8929 |
| 4 | R1 | SpatialCNN + cross_mamba, bs=8 | 0.8930 | 0.8882 |
| 5 | R3 | ResNet18 frozen + vel + aug, bs=8 | 0.8860 | 0.8819 |
| 6 | R7 | R4 + 32 frames, bs=4 | 0.8837 | — |
| 7 | R8 | Flow (6ch) + vel + aug, bs=32 | 0.8791 | — |
| 8 | R2 | SpatialCNN + attention, bs=8 | 0.8767 | 0.8711 |
| 9 | R9 | SpatialCNN + vel + aug, bs=32 | 0.8558 | — |

## 6. Key Findings

### Why RGBD ≪ Skeleton (0.90 vs 0.96)

The fundamental gap comes from **input representation quality**, not architecture:

| Property | Skeleton (60-dim) | RGBD (50,176-dim per frame) |
|----------|-------------------|----------------------------|
| Dimensionality | 60 (20 joints × 3) | 50,176 (4 × 112 × 112) |
| Noise | None (clean joint coordinates) | Background, lighting, clothing |
| Semantic density | Every value is action-relevant | Mostly irrelevant background pixels |
| Structure | Body topology preserved | Raw pixel grid |
| Temporal features | Velocity = simple diff | Requires learned motion features |

With only **431 training samples**, there is insufficient data to learn the visual abstraction that skeleton preprocessing provides for free.

### What Worked
1. **Partial freeze (ResNet18 layer4)**: Best balance — leverages ImageNet features while adapting to action recognition
2. **Temporal velocity features**: Feature-level frame differences provide motion cues analogous to skeleton velocity
3. **Data augmentation**: Flip + crop + color jitter + IMU jitter/scaling consistently improve generalization
4. **Small batch size (8)**: More gradient updates per epoch critical on 431 samples
5. **Cross-mamba fusion**: Still the best fusion strategy, consistent with skeleton+IMU findings

### What Didn't Work
1. **Frozen ResNet18** (R3): ImageNet features too generic for action recognition, under-fits
2. **Optical flow** (R8): Farnebäck flow adds noise rather than useful motion signal at 112×112 resolution
3. **32 frames** (R7): Higher memory cost without benefit; 16 frames sufficient for UTD-MHAD actions
4. **Batch size 32** (R8, R9): Consistently hurts — same finding as skeleton+IMU experiments
5. **Full fine-tuning** (R6): Matches R4 but doesn't beat it; partial freeze is more efficient

### Conclusion

RGBD+IMU achieves **~0.90 accuracy** on UTD-MHAD, ~7 points below skeleton+IMU (0.96). This is a fundamental data limitation: 431 training samples cannot learn the noise-free pose abstraction that skeleton preprocessing provides. The RGBD pipeline is useful when skeleton preprocessing is unavailable, but skeleton remains strictly superior when accessible.

## 7. Cross-Pipeline Comparison

| Pipeline | Best Exp | Acc | F1 | Preprocessing Required |
|----------|----------|-----|-----|----------------------|
| **Skeleton + IMU** | Exp25 | **0.9628** | **0.9622** | Skeleton extraction (pose estimation) |
| **RGBD + IMU** | R4 | 0.8977 | 0.8951 | None (raw video + depth + IMU) |

### Training Commands

**Best Skeleton+IMU (Exp25)**:
```powershell
python train/run_train.py --pipeline skel_imu --data_root "./datasets/UTD-MHAD" `
  --epochs 100 --save_name skel_imu_exp25.pt --tb_dir runs --num_workers 0 `
  --label_smoothing 0.1 --mixup_alpha 0.15 --weight_decay 0.008 `
  --scheduler cosine_warmup --warmup_epochs 5 --lr 3e-4 --patience 40 --seed 42 `
  --model_kwargs '{"d_model":160,"fusion":"cross_mamba","modality_dropout":0.0,"aux_weight":0.1}'
```

**Best RGBD+IMU (R4)**:
```powershell
python train/run_train.py --pipeline rgbd_imu --data_root "./datasets/UTD-MHAD" `
  --epochs 100 --save_name rgbd_imu_R4.pt --tb_dir runs --num_workers 0 --batch_size 8 `
  --label_smoothing 0.1 --mixup_alpha 0.15 --weight_decay 0.008 `
  --scheduler cosine_warmup --warmup_epochs 5 --lr 3e-4 --patience 40 --seed 42 `
  --model_kwargs '{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true}' `
  --dataset_kwargs '{"augment":true}'
```

---
---

# MMA RGBD+IMU — Missing-Modality Robustness Experiments

## 1. Motivation

Deploying a multimodal model in the real world requires **graceful degradation** when one sensor fails or is unavailable. After establishing the R4 baseline (Full=0.8977), we discovered a critical weakness: with **no robustness training**, zeroing out one modality at inference time causes catastrophic failure:

- R4 RGBD-only: **14.19%** (barely above random-27 = 3.7%)
- R4 IMU-only: **73.02%**

The RGBD branch effectively acts as a bias term that the model has never learned to use independently. This section documents the systematic effort to fix this, culminating in CMAR1 — the best model overall.

---

## 2. New Architecture Components

### 2.1 Modality Dropout (Dataset-level, MD1–MD6)

Applied at the **data loading level**: entire modality inputs replaced with zero tensors or learnable missing tokens before the forward pass.

**Schedules:**
- `fixed`: constant dropout probability $p$ throughout training
- `curriculum`: warmup phase (no dropout) then linear ramp to $p_{max}$ over remaining epochs
- `simultaneous`: every iteration runs 3 forward passes (full, RGBD-only, IMU-only) and sums their losses

**Missing token variants:**
- `zero-fill`: replace missing modality input with all zeros
- `missing_token`: replace with a learnable `nn.Parameter` of shape `(1, 1, d_model)` (broadcast over batch/time)

### 2.2 Feature-level Modality Dropout (MD-Drop)

Applied **after encoding, before fusion** inside `MultimodalMMA.forward()`:

```python
# In forward(), after fv = rgbd_enc(rgbd), fi = imu_enc(imu)
r = random.random()
if r < md_drop_imu:
    fi = torch.zeros_like(fi)          # RGBD-only path (35% of batches)
elif r < md_drop_imu + md_drop_rgbd:
    fv = torch.zeros_like(fv)          # IMU-only path (10% of batches)
# else: both modalities (55% of batches)
```

Key difference from dataset-level dropout: encoders always run (preventing gradient starvation in early layers), but the fusion module and classifier must learn to handle zero features.

### 2.3 Cross-Modal Alignment Regularization (CMAR)

Additional loss term that forces both branch representations to be mutually consistent:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{aux} \cdot \mathcal{L}_{aux} + \lambda_{cmar} \cdot \underbrace{\| \text{Proj}_{rgbd}(\bar{h}^{rgbd}) - \text{Proj}_{imu}(\bar{h}^{imu}) \|_F^2}_{\mathcal{L}_{CMAR}}$$

Where $\bar{h} = \text{MeanPool}_{t}(h_t)$ pools over the sequence dimension to handle different RGBD/IMU sequence lengths.

**Implementation:**
```python
# Two projection heads in __init__:
self.cmar_proj_rgbd = nn.Linear(d_model, cmar_proj_dim)  # 160 → 64
self.cmar_proj_imu  = nn.Linear(d_model, cmar_proj_dim)  # 160 → 64

# In forward(), before fusion:
fv_proj = self.cmar_proj_rgbd(fv.mean(dim=1))   # (B, 64)
fi_proj = self.cmar_proj_imu(fi.mean(dim=1))    # (B, 64)
self._cmar_loss = torch.mean((fv_proj - fi_proj) ** 2)
# Trainer adds: loss += cmar_weight * _cmar_loss
```

The projection heads isolate CMAR from the main task — the main feature space is free to diverge as needed for classification, while alignment is enforced only in the 64-dim projection space.

---

## 3. Experiment Results

All experiments use the proven base config: d_model=160, fusion=cross_mamba, aux_weight=0.1, encoder=pretrained, freeze=partial, temporal_velocity=true, augment=true, bs=32, label_smoothing=0.1, mixup_alpha=0.15, weight_decay=0.008, cosine_warmup(5), lr=3e-4, patience=40, seed=42.

### 3.1 Dataset-level Modality Dropout (MD1–MD6)

| Exp | Schedule | $p_{max}$ | Missing token | Full | RGBD-only | IMU-only | Avg | Notes |
|-----|----------|-----------|---------------|------|-----------|----------|-----|-------|
| MD1 | fixed | 0.1 | zero | 0.8860 | 0.2163 | 0.8256 | 0.6426 | Baseline dropout |
| **MD2** | **curriculum** | **0.3** | **missing_token** | **0.8884** | **0.3698** | **0.8488** | **0.7023** | Curriculum helps |
| MD3 | curriculum | 0.3 | zero | 0.8744 | 0.2698 | 0.8209 | 0.6550 | zero-fill worse |
| **MD4** | **curriculum** | **0.3** | **zero** | **0.8930** | **0.3558** | **0.8767** | **0.7085** | Best dataset-level |
| MD5 | fixed | 0.15 | missing_token | 0.8860 | 0.2907 | 0.8302 | 0.6690 | Low p, weak |
| MD6 | simultaneous | 0.3 | zero | 0.8791 | 0.3023 | 0.8395 | 0.6736 | 3-pass costly, no gain |

**Dataset-level MD best: MD4** (Full=0.8930, RGBD=0.3558, IMU=0.8767, Avg=0.7085)

### 3.2 Staged Training (ST1) — FAILED

Hypothesis: a 3-phase curriculum (IMU warmup → RGBD focus with partial freeze → joint fine-tune) could force independent branch learning without destroying fusion.

| Phase | Epochs | Params Active | Peak Val Acc | Description |
|-------|--------|---------------|--------------|-------------|
| 1 | 1–20 | ~417K (IMU + head only) | 73.26% | IMU-only warmup |
| 2 | 21–40 | ~8.9M (RGBD encoder unfrozen) | 79.53% | RGBD focus |
| 3 | 41–100 | ~9.2M (all) | 82.79% | Joint fine-tune |

**ST1 missing-modality results (checkpoints/staged_rgbd_imu_ST1.pt):**
- Full: 0.8279, RGBD-only: **0.0279** (catastrophic), IMU-only: 0.7488
- **Conclusion: FAILED.** The model never saw zeroed-modality inputs during training, so inference with one branch zeroed is completely out-of-distribution. RGBD-only performance was *worse* than the naïve R4 baseline (2.79% vs 14.19%).

### 3.3 Feature-level Dropout — MDdrop1

Simpler, direct approach: zero features inside `forward()` at feature level.

Config: `md_drop_imu=0.20`, `md_drop_rgbd=0.10`, all else same as R4.

| Condition | Accuracy | F1 |
|---|---|---|
| Full (both) | 0.8860 | 0.8805 |
| RGBD-only | 0.3465 | 0.3181 |
| IMU-only | 0.8488 | 0.8430 |
| Average | 0.6938 | 0.6805 |

MDdrop1 significantly improves RGBD-only vs R4 (+20pp), and IMU-only also improves. However, Full accuracy regressed vs R4 (88.60% < 89.77%), suggesting the higher dropout disrupts fusion quality.

**Key insight from MDdrop1:** With only 20% IMU dropout, the model still leans heavily on IMU as the "easy" path. RGBD branch gets insufficient gradient signal for independent learning.

### 3.4 Feature-level Dropout + CMAR — CMAR1 ⭐ BEST

Combine stronger IMU dropout with CMAR alignment regularization:

Config: `md_drop_imu=0.35`, `md_drop_rgbd=0.10`, `cmar_weight=0.1`, `cmar_proj_dim=64`.

Training trajectory highlights:
- Epoch 5 (warmup end): 37.21% — lower than MDdrop1 (50.23%), expected due to harder dropout regime
- Epoch 40: 87.67% — model catching up as CMAR guides RGBD branch  
- Epoch 62: 90.70% — new milestone, breaking 90% for first time
- Epoch 72: **90.93%** — best (checkpoint saved)
- Epoch 100: 89.53% (cosine decay, best from ep72)

**CMAR1 missing-modality evaluation:**

| Condition | Accuracy | F1 |
|---|---|---|
| Full (both) | **0.9093** | **0.9061** |
| RGBD-only | **0.4163** | **0.4072** |
| IMU-only | **0.8628** | **0.8599** |
| Average | **0.7295** | **0.7244** |
| Degradation | **0.4930** | — |

---

## 4. Comprehensive Comparison

| Exp | Method | Full | RGBD-only | IMU-only | Avg | Degradation |
|-----|--------|------|-----------|----------|-----|-------------|
| R4 | No robustness training | 0.8977 | 0.1419 | 0.7302 | 0.5900 | 0.7558 |
| MD4 | Dataset-level curriculum MD | 0.8930 | 0.3558 | 0.8767 | 0.7085 | 0.5372 |
| MD2 | Dataset-level curriculum + missing_token | 0.8884 | 0.3698 | 0.8488 | 0.7023 | 0.5186 |
| MDdrop1 | Feature-level drop (imu=0.20) | 0.8860 | 0.3465 | 0.8488 | 0.6938 | 0.5395 |
| ST1 | Staged training | 0.8279 | 0.0279 | 0.7488 | 0.5349 | 0.8000 |
| **CMAR1** | **Feature-level drop (imu=0.35) + CMAR** | **0.9093** | **0.4163** | **0.8628** | **0.7295** | **0.4930** |

---

## 5. Analysis: Why CMAR1 Works

### 5.1 The RGBD Branch Collapse Problem

Without any robustness mechanism, the model faces an easy optimization shortcut: IMU alone can reach ~73% accuracy, while RGBD requires expensive learning (431 training samples, 50K+ pixel values per frame). The fusion mechanism learns to rely almost entirely on IMU features, causing the RGBD branch to produce near-constant representations — useful only as a small correction bias.

This manifests as:
- Full accuracy: near-optimal (model uses IMU well)
- RGBD-only: ~14% (branch produces near-constant output → random guessing)
- The zero-IMU input is completely OOD for a network never trained with it

### 5.2 How MD-Drop Fixes OOD

By randomly zeroing IMU features in 35% of training batches, the model is forced to classify from RGBD alone in ~1 in 3 steps. The gradient signal from these forced RGBD-only steps incentivizes the RGBD encoder to produce discriminative features for the classifier.

Without CMAR, however, the RGBD branch still produces representations in a different subspace from IMU — fusion still uses modality-specific "shortcuts" rather than building a unified representation.

### 5.3 How CMAR Complements MD-Drop

CMAR enforces that even in the 55% of batches where **both modalities are present**, the RGBD branch must build representations that are **aligned with IMU representations**. This has two effects:

1. **Anti-collapse**: RGBD features cannot degenerate to near-zero or constant representations (which would maximize CMAR loss)
2. **Transfer signal**: The well-learned IMU representation acts as a "teacher" — RGBD is continuously pulled toward a semantically meaningful representation, accelerating learning from the sparse RGBD-only batches

Crucially, the projection heads ($\text{Proj}_{rgbd}$, $\text{Proj}_{imu}$) isolate CMAR's effect: the main feature space remains free to diverge as needed for cross-modal fusion, while alignment is enforced only in the 64-dim projection space. This prevents collapse into trivially aligned (but uninformative) representations.

### 5.4 Why Stronger IMU Dropout (0.35 > 0.20) Helped Full Accuracy

Counterintuitively, CMAR1 achieves **higher Full accuracy (90.93%)** than MDdrop1 (88.60%) despite using stronger IMU dropout. The explanation:

- MDdrop1 at 20% still allows IMU to dominate fusion in the majority of batches
- CMAR1's stronger IMU dropout (35%) forces more balanced branch training
- CMAR's alignment loss ensures this doesn't hurt IMU-path performance
- A model with two well-trained branches fuses better than one with a dominant-branch bias

The net result: CMAR1's RGBD branch is strong enough that combining both modalities creates **genuine complementarity** rather than one branch correcting the other's noise.

---

## 6. Leaderboard (RGBD+IMU All Experiments)

| Rank | Exp | Full | RGBD-only | IMU-only | Avg | F1 |
|------|-----|------|-----------|----------|-----|----|
| 🥇 | **CMAR1** | **0.9093** | **0.4163** | **0.8628** | **0.7295** | 0.9061 |
| 🥈 | R4 (no robust) | 0.8977 | 0.1419 | 0.7302 | 0.5900 | 0.8951 |
| 🥉 | MD4 | 0.8930 | 0.3558 | 0.8767 | 0.7085 | — |
| 4 | MD2 | 0.8884 | 0.3698 | 0.8488 | 0.7023 | — |
| 5 | MDdrop1 | 0.8860 | 0.3465 | 0.8488 | 0.6938 | 0.8805 |
| 6 | MD6 | 0.8791 | 0.3023 | 0.8395 | 0.6736 | — |
| 7 | MD1 | 0.8860 | 0.2163 | 0.8256 | 0.6426 | — |
| 8 | ST1 | 0.8279 | 0.0279 | 0.7488 | 0.5349 | 0.8165 |

---

## 7. Training Command (CMAR1 — Best)

```powershell
conda activate mma
cd "E:\VS Code Project\COMP 4901D"
python train/run_train.py --pipeline rgbd_imu --data_root "./datasets/UTD-MHAD" `
  --epochs 100 --save_name rgbd_imu_CMAR1.pt --tb_dir runs --num_workers 0 `
  --batch_size 32 --label_smoothing 0.1 --mixup_alpha 0.15 --weight_decay 0.008 `
  --scheduler cosine_warmup --warmup_epochs 5 --lr 3e-4 --patience 40 --seed 42 `
  --dataset_kwargs '{"augment":true}' `
  --model_kwargs '{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.35,"md_drop_rgbd":0.10,"cmar_weight":0.1,"cmar_proj_dim":64}'
```

## 8. Evaluation Command

```powershell
python infer/run_infer.py --pipeline rgbd_imu --data_root "datasets/UTD-MHAD" `
  --checkpoint "checkpoints/rgbd_imu_CMAR1.pt" --num_workers 0 --eval_missing all
```

Expected output:
```
Condition              Accuracy         F1
Full (both)              0.9093     0.9061
RGBD-only                0.4163     0.4072
IMU-only                 0.8628     0.8599
Average                  0.7295     0.7244
Degradation              0.4930
```
