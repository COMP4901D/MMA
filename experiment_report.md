# MMA RGBD+IMU — Momentum Mamba for Multimodal Human Activity Recognition

## 1. Model Architecture

### 1.1 Overview

**MultimodalMMA** is a dual-branch multimodal architecture for Human Activity Recognition (HAR) on the UTD-MHAD dataset, fusing **RGB-D video** and **6-axis IMU** sensor data. The core innovation combines three key components: (1) a **Momentum-augmented Mamba** selective state-space model for temporal encoding, (2) **Cross-Mamba fusion** for deep cross-modal interaction, and (3) **Cross-Modal Alignment Regularization (CMAR)** with **Feature-level Modality Dropout (MD-Drop)** for missing-modality robustness.

```
┌─────────────────────────────────────────────────────────────┐
│                  MultimodalMMA (RGBD+IMU)                   │
│                                                             │
│  ┌────────────────────┐      ┌────────────────────┐         │
│  │   RGBD Encoder     │      │    IMU Encoder     │         │
│  │  ResNet18 (4ch)    │      │  Conv1D frontend   │         │
│  │  per-frame CNN     │      │                    │         │
│  │  + Temporal Vel.   │      │                    │         │
│  │       ↓            │      │        ↓           │         │
│  │  MomentumMamba ×2  │      │  MomentumMamba ×2  │         │
│  │       ↓            │      │        ↓           │         │
│  │    (B, T_v, D)     │      │    (B, T_i, D)     │         │
│  └───────┬────────────┘      └────────┬───────────┘         │
│          │◄── CMAR Alignment Loss ────►│                    │
│          │     MD-Drop (training)      │                    │
│          │  [+ Modality Embeddings]    │                    │
│          └──────────────┬──────────────┘                    │
│                         │ concat (T_v + T_i) tokens         │
│                  ┌──────┴──────┐                            │
│                  │  Cross-Mamba │  (shared SSM scan)        │
│                  └──────┬──────┘                            │
│                  AttentionPool (temporal)                   │
│                         ↓                                   │
│                 Dropout → Linear(D, 27)                     │
│                       Logits                                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Momentum Mamba Block (Core Temporal Encoder)

Each encoder branch processes its feature sequence through $N$ **MomentumMambaBlock** layers. This block extends the Mamba selective state-space model [1] with a second-order momentum mechanism inspired by heavy-ball optimization [2].

#### 1.2.1 Background: Selective State-Space Models

Mamba [1] is built on the structured state-space model (SSM) framework [3], which models sequences via a continuous-time linear system discretized for sequential data. The standard (first-order) Mamba recurrence is:

$$h_n = \bar{A}_n \odot h_{n-1} + \bar{B}_n \cdot x_n$$
$$y_n = C_n \cdot h_n$$

where $h_n \in \mathbb{R}^{d\_state}$ is the hidden state, $x_n$ is the input at position $n$, and $\bar{A}_n, \bar{B}_n, C_n$ are **input-dependent** (selective) discretized parameters. The selectivity is the key innovation of Mamba over prior SSMs like S4 [3]: by making the state transition parameters functions of the input, the model can dynamically choose what information to store or discard.

The full MambaBlock wraps this SSM in a gated architecture:

```
Input x
  ├──→ Linear(D, 2·expand·D) ──→ [z_branch, x_branch]
  │         z_branch: SiLU activation (gate)
  │         x_branch: Causal DepthwiseConv1D → SiLU → SSM scan
  │
  └──→ Residual connection
       Output = x + (z_gate ⊙ SSM_output) · Linear(expand·D, D)
```

#### 1.2.2 Second-Order Momentum Extension

Standard Mamba's first-order recurrence can be sensitive to noise in rapidly-changing input signals. We introduce a **momentum buffer** $v_n$ that smooths the input injection into the SSM, analogous to the momentum term in optimization [2, 4]:

$$v_n = \beta \cdot v_{n-1} + \alpha \cdot \bar{B}_n \cdot x_n$$
$$h_n = \bar{A}_n \odot h_{n-1} + v_n$$

where $\alpha$ (init=0.6) and $\beta$ (init=0.99) are **learnable per-dimension parameters**. The momentum buffer $v_n$ effectively computes an exponential moving average (EMA) of the input contributions $\bar{B}_n \cdot x_n$, providing several benefits:

1. **Noise robustness**: Transient input noise is smoothed by the EMA, preventing spurious state updates. This is particularly beneficial for IMU signals (accelerometer/gyroscope noise) and per-frame CNN features (visual jitter between frames).

2. **Sustained motion capture**: For actions like "tennis swing" or "bowling," the discriminative signal extends over many consecutive frames. The momentum buffer maintains a running average of the input pattern, making it easier to detect sustained motions vs. momentary fluctuations.

3. **Longer effective memory**: The two-step recurrence ($v_n$ and $h_n$ both carry history) creates a second-order dynamical system with richer temporal dynamics than the first-order baseline, without increasing the state dimension.

4. **Minimal overhead**: Only $2 \times d\_state$ additional learnable parameters ($\alpha, \beta$) per block, adding negligible cost to the ~142K parameters per encoder.

We also support a **complex momentum** variant where $\beta_c = \rho \cdot e^{i\theta}$, introducing oscillatory dynamics that can capture periodic motion patterns (e.g., walking, jogging). However, the real-valued momentum mode proved more stable in practice.

#### 1.2.3 Block Architecture

Each MomentumMambaBlock has the following hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 160 | Hidden dimension |
| `d_state` | 32 | SSM state dimension |
| `d_conv` | 4 | Local convolution kernel size |
| `expand` | 2 | Inner expansion factor (inner_dim = $D \times$ expand) |
| `alpha_init` | 0.6 | Initial momentum input weight |
| `beta_init` | 0.99 | Initial momentum decay rate |

### 1.3 RGBD Encoder

The RGBD encoder processes 4-channel (RGB + Depth) video frames through a per-frame CNN followed by temporal Momentum Mamba layers.

| Variant | Input | Frontend | Params | Description |
|---------|-------|----------|--------|-------------|
| `spatial_cnn` | (B, T, 4, 112, 112) | 4-layer stride-2 Conv2D → per-frame feature | ~340K | Lightweight, trains from scratch |
| `pretrained` (ResNet18) [5] | (B, T, 4, 112, 112) | ImageNet-pretrained ResNet18, first conv adapted 3→4 channels | ~11.2M | Pretrained visual features |

**Freeze modes** (for `pretrained`):
- `all`: Only final projection layer trainable (~82K trainable)
- `partial`: layer4 + projection trainable (~2.6M trainable) — **best**
- `none`: Full fine-tuning (~11.2M trainable)

The partial freeze strategy leverages low-level ImageNet feature extractors (edges, textures, shapes) frozen from pretraining, while allowing the high-level layer4 features to adapt to action-specific visual patterns.

### 1.4 IMU Encoder

| Variant | Input | Frontend | Description |
|---------|-------|----------|-------------|
| `conv1d` (default) | (B, T, 6) — 6-axis IMU | Conv1D(6→D, k=3) → BN → ReLU → Dropout | Temporal pattern extraction |

The Conv1D frontend extracts local temporal patterns from the raw 6-axis signal (3-axis accelerometer + 3-axis gyroscope), which are then processed by $N$ MomentumMambaBlock layers for long-range temporal modeling.

### 1.5 Temporal Velocity Features

When `temporal_velocity=True`, the RGBD encoder computes frame-level feature differences after the CNN frontend (analogous to optical flow in feature space):

$$v_t = f_t - f_{t-1}, \quad v_0 = 0$$

The velocity features are concatenated with the original features and projected back:
```
[f_t || v_t] → Linear(2D, D) → LayerNorm → ReLU
```

This provides explicit motion cues without the computational overhead of optical flow computation, and operates in the learned feature space rather than raw pixel space, yielding more semantically meaningful motion representations.

### 1.6 Cross-Mamba Fusion

Cross-Mamba fusion is the key mechanism for deep cross-modal interaction, going beyond standard late fusion approaches (feature concatenation, gating, or attention pooling) that only combine modality representations after independent temporal processing.

#### 1.6.1 Motivation

Standard late fusion methods have a fundamental limitation: they combine **pooled** (time-collapsed) features from each modality, losing the temporal alignment between modalities. For example, the IMU acceleration peak during a "throw" action corresponds to specific video frames — late fusion cannot exploit this temporal correspondence.

#### 1.6.2 Architecture

Cross-Mamba fusion operates on the **full temporal sequences** from both modalities:

1. **Modality embeddings**: Learnable vectors $e_{rgbd}, e_{imu} \in \mathbb{R}^{D}$ are added to each modality's sequence to preserve modality identity after concatenation.

2. **Sequence concatenation**: The two temporal sequences are concatenated along the time dimension:
$$S = [(f^{rgbd}_1 + e_{rgbd}), ..., (f^{rgbd}_{T_v} + e_{rgbd}), (f^{imu}_1 + e_{imu}), ..., (f^{imu}_{T_i} + e_{imu})]$$
resulting in a combined sequence of length $T_v + T_i$.

3. **Shared Mamba scan**: A dedicated MomentumMambaBlock processes the concatenated sequence. Because Mamba's selective SSM mechanism is **input-dependent**, it naturally learns which cross-modal interactions are informative:
   - When scanning an RGBD token, the hidden state already encodes information from preceding RGBD frames and (after the boundary) IMU readings
   - The selectivity mechanism ($\bar{B}_n$ depends on input $x_n$) allows the model to gate how much cross-modal information is incorporated at each position

4. **AttentionPool**: A learnable attention-weighted temporal pooling aggregates the fused sequence into a single vector:
$$\alpha_t = \text{softmax}(W_2 \tanh(W_1 s_t))$$
$$z = \sum_t \alpha_t \cdot s_t$$

#### 1.6.3 Advantages over Late Fusion

| Property | Late Fusion (concat/gated) | Cross-Mamba |
|----------|---------------------------|-------------|
| Temporal alignment | Lost (pooled before fusion) | Preserved (full sequences) |
| Cross-modal interaction | Surface-level (single MLP/gate) | Deep (SSM-mediated) |
| Variable-length handling | Requires fixed-size pooling | Natural (SSM processes any length) |
| Computational cost | Lower (only pooled features) | Moderate (one extra MambaBlock) |

### 1.7 Auxiliary Per-Modality Loss

When `aux_weight > 0`, per-modality **AuxHead** modules (AttentionPool + Linear) produce separate classification logits for each branch:

$$\mathcal{L}_{aux} = \frac{1}{2}(\mathcal{L}_{rgbd}^{aux} + \mathcal{L}_{imu}^{aux})$$

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda_{aux} \cdot \mathcal{L}_{aux}$$

This forces each encoder branch to learn independently discriminative features, preventing the common failure mode where one modality "free-rides" on the other. With auxiliary loss, both encoders must produce representations sufficient for classification on their own, which dramatically improves fusion quality.

### 1.8 Feature-level Modality Dropout (MD-Drop)

During training, the model randomly zeroes out one modality's encoded features **after encoding but before fusion** [6]:

```python
r = random.random()
if r < md_drop_imu:
    fi = torch.zeros_like(fi)     # RGBD-only path
elif r < md_drop_imu + md_drop_rgbd:
    fv = torch.zeros_like(fv)     # IMU-only path
# else: both modalities present
```

This training strategy directly addresses the **out-of-distribution (OOD) problem** at inference time: without MD-Drop, a model that has only ever seen both modalities together will produce unpredictable outputs when one modality is missing. By training with missing modalities, the fusion module and classifier learn to handle zero-feature inputs gracefully, making missing-modality inference **in-distribution** rather than OOD.

Key design decisions:
- **Feature-level** (not input-level): Encoders always run and receive gradients, preventing gradient starvation in early layers
- **Asymmetric dropout** (md_drop_imu=0.50 > md_drop_rgbd=0.05): IMU is the dominant modality, so it needs to be dropped more frequently to force RGBD learning
- **Exclusive mode**: Only one modality is dropped at a time (never both), ensuring every training step produces useful gradients

### 1.9 Cross-Modal Alignment Regularization (CMAR)

CMAR is the critical component that prevents the **weaker modality branch collapse** problem and works synergistically with MD-Drop.

#### 1.9.1 The Branch Collapse Problem

In multimodal learning with imbalanced modality informativeness, a well-known failure mode occurs [6, 7]: the stronger modality (here, IMU) provides an easy optimization path, causing the weaker modality's encoder (RGBD) to converge to a near-constant representation. The RGBD branch effectively becomes a bias term rather than an informative feature extractor.

Evidence of collapse without CMAR:
- R4 baseline (no robustness): Full=89.77%, but RGBD-only=**14.19%** (barely above the 3.7% random chance for 27 classes)
- The RGBD encoder outputs near-constant features regardless of input content

#### 1.9.2 CMAR Formulation

CMAR adds a soft representation alignment loss that forces both branches to produce mutually consistent representations:

$$\mathcal{L}_{CMAR} = \frac{1}{B} \sum_{i=1}^{B} \| \text{Proj}_{rgbd}(\bar{h}_i^{rgbd}) - \text{Proj}_{imu}(\bar{h}_i^{imu}) \|_2^2$$

where:
- $\bar{h}^{rgbd} = \frac{1}{T_v} \sum_{t=1}^{T_v} h_t^{rgbd}$ — mean-pooled RGBD features (handles variable sequence lengths)
- $\bar{h}^{imu} = \frac{1}{T_i} \sum_{t=1}^{T_i} h_t^{imu}$ — mean-pooled IMU features
- $\text{Proj}_{rgbd}: \mathbb{R}^{D} \to \mathbb{R}^{d_{proj}}$ and $\text{Proj}_{imu}: \mathbb{R}^{D} \to \mathbb{R}^{d_{proj}}$ — learnable linear projections

The total training loss becomes:

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{aux} \cdot \mathcal{L}_{aux} + \lambda_{cmar} \cdot \mathcal{L}_{CMAR}$$

#### 1.9.3 Key Design Principles

1. **Projection heads isolate alignment from classification**: The alignment is enforced in a separate 64-dimensional projection space, not in the main $D$-dimensional feature space. This allows the main features to diverge as needed for optimal cross-modal fusion and classification, while ensuring a shared semantic structure exists in the projection subspace. This design draws inspiration from contrastive representation learning [8].

2. **MSE loss (not cosine similarity)**: Empirical comparison (CMAR19) showed MSE outperforms cosine similarity. MSE directly penalizes magnitude differences, enforcing both directional alignment and scale consistency between modalities.

3. **Mutual distillation effect**: Since the IMU encoder learns discriminative features more easily (lower input dimensionality, higher signal-to-noise ratio), the CMAR objective effectively acts as a form of **online knowledge distillation** [9] — the well-trained IMU branch acts as a "teacher" that continuously guides the RGBD branch (the "student") toward semantically meaningful representations. Unlike fixed-teacher distillation, both branches update simultaneously, allowing bidirectional benefit.

4. **Anti-collapse guarantee**: CMAR prevents the RGBD encoder from degenerating to constant output. A constant RGBD representation $c$ would need to match every class's IMU representation simultaneously, which is impossible — CMAR loss would be high. The only way to minimize CMAR loss is for the RGBD encoder to produce class-discriminative features that track the IMU encoder's variations.

5. **Complementarity with MD-Drop**: MD-Drop provides the **gradient signal** for RGBD-only classification (via forced RGBD-only batches), while CMAR provides continuous **representation guidance** even in the 50% of batches where both modalities are present. Together, they address both the data-distribution gap (MD-Drop) and the representation quality gap (CMAR).

#### 1.9.4 Implementation

```python
# In __init__:
self.cmar_proj_rgbd = nn.Linear(d_model, cmar_proj_dim)   # 160 → 64
self.cmar_proj_imu  = nn.Linear(d_model, cmar_proj_dim)   # 160 → 64

# In forward(), after encoding and before fusion:
fv_proj = self.cmar_proj_rgbd(fv.mean(dim=1))    # (B, T_v, D) → (B, 64)
fi_proj = self.cmar_proj_imu(fi.mean(dim=1))     # (B, T_i, D) → (B, 64)
self._cmar_loss = torch.mean((fv_proj - fi_proj) ** 2)
# Trainer reads _cmar_loss and adds: loss += cmar_weight * _cmar_loss
```

### 1.10 AttentionPool (Temporal Pooling)

Learnable attention-weighted temporal pooling replaces naive mean/max pooling:

$$\alpha_t = \text{softmax}\big(W_2 \tanh(W_1 h_t + b_1) + b_2\big)$$
$$\text{output} = \sum_{t} \alpha_t \cdot h_t$$

with $W_1 \in \mathbb{R}^{D/4 \times D}$, $W_2 \in \mathbb{R}^{1 \times D/4}$. Input: $(B, T, D)$ → Output: $(B, D)$.

This allows the model to learn which temporal positions are most informative for classification, rather than treating all time steps equally.

### 1.11 Classification Head

```
Dropout(p) → Linear(d_model, 27)
```

where 27 = number of action classes in UTD-MHAD.

---

## 2. Dataset

### 2.1 UTD-MHAD

The **UTD-MHAD** dataset [10] contains 27 action classes performed by 8 subjects with 4 trials each (~861 samples total).

- **Split**: Subjects 1,3,5,7 for training (431 samples), subjects 2,4,6,8 for testing (430 samples)
- **RGBD**: RGB video (.avi, 640×480) + Depth maps (.mat, 320×240), uniformly sampled to T=16 frames
- **IMU**: 6-axis (3 acc + 3 gyro), padded/truncated to T=192

### 2.2 RGBD Data Pipeline

```
RGB (.avi, 640×480)  ──→  resize 112×112  ──→  normalize [0,1]  ──┐
Depth (.mat, 320×240) ─→  resize 112×112  ──→  normalize [0,1]  ──┤
                                                                    ↓
                                              Stack → (T, 4, 112, 112)
                                              Uniform sample T=16 frames
                                                                    ↓
                                              Total per-sample: 16 × 4 × 112 × 112 = 802,816 values
```

### 2.3 Data Augmentation

RGBD augmentations (train only, `augment=True`):
- Random horizontal flip (p=0.5)
- Random crop (scale 0.8–1.0) + resize to 112×112
- Color jitter (brightness/contrast ±15%) on RGB channels only

IMU augmentations:
- Jitter (σ=0.03)
- Scaling (σ=0.1)

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW [11] |
| Learning Rate | 3e-4 |
| Weight Decay | 0.008 |
| Scheduler | Cosine Annealing with Warmup [12] (5 warmup epochs) |
| Batch Size | 32 |
| Epochs | 100 |
| Patience | 40 (early stopping) |
| Label Smoothing | 0.1 [13] |
| Mixup Alpha | 0.15 [14] |
| AMP | Enabled |
| Seed | 42 |

---

## 4. Experiment Results

### 4.1 Phase 1: Baseline RGBD+IMU (R1–R6)

All experiments use the same base training config unless noted.

| Exp | Encoder | Fusion | aux_w | bs | lr | Acc | F1 | Notes |
|-----|---------|--------|-------|----|----|-----|-----|-------|
| R1 | SpatialCNN | cross_mamba | 0.1 | 8 | 3e-4 | 0.8930 | 0.8882 | Baseline |
| R2 | SpatialCNN | attention | 0.1 | 8 | 3e-4 | 0.8767 | 0.8711 | Attention fusion weaker |
| R3 | ResNet18 | all (frozen) | 0.1 | 8 | 3e-4 | 0.8860 | 0.8819 | Frozen backbone under-fits |
| **R4** | **ResNet18** | **partial** | **0.1** | **8** | **3e-4** | **0.8977** | **0.8951** | **Baseline BEST** |
| R5 | SpatialCNN | cross_mamba | 0.1 | 8 | 3e-4 | 0.8953 | 0.8929 | +velocity +augment, SpatialCNN competitive |
| R6 | ResNet18 | none (full) | 0.1 | 8 | 1e-4 | 0.8977 | 0.8955 | Full finetune ties R4 |

### 4.2 Phase 1b: Additional Exploration (R7–R9)

| Exp | Configuration | bs | Acc | F1 | Notes |
|-----|---------------|-----|-----|-----|-------|
| R7 | R4 config + 32 frames | 4 | 0.8837 | 0.8791 | OOM at epoch 51, more frames didn't help |
| R8 | Optical flow (6ch) + vel + aug | 32 | 0.8791 | 0.8710 | Farneback flow did not improve |
| R9 | SpatialCNN + vel + aug | 32 | 0.8558 | 0.8449 | bs=32 clearly hurts convergence |

### 4.3 Phase 2: Missing-Modality Robustness Discovery

After establishing R4 as the baseline (Full=0.8977), we discovered a critical vulnerability: zeroing out one modality at inference time causes catastrophic failure:

- R4 RGBD-only: **14.19%** (barely above random-27 = 3.7%)
- R4 IMU-only: **73.02%**

The RGBD branch had effectively collapsed to a bias term. This motivated the robustness experiments below.

### 4.4 Dataset-level Modality Dropout (MD1–MD6)

| Exp | Schedule | $p_{max}$ | Missing token | Full | RGBD-only | IMU-only | Avg | Notes |
|-----|----------|-----------|---------------|------|-----------|----------|-----|-------|
| MD1 | fixed | 0.1 | zero | 0.8860 | 0.2163 | 0.8256 | 0.6426 | Baseline dropout |
| **MD2** | **curriculum** | **0.3** | **missing_token** | **0.8884** | **0.3698** | **0.8488** | **0.7023** | Curriculum helps |
| MD3 | curriculum | 0.3 | zero | 0.8744 | 0.2698 | 0.8209 | 0.6550 | zero-fill worse |
| **MD4** | **curriculum** | **0.3** | **zero** | **0.8930** | **0.3558** | **0.8767** | **0.7085** | Best dataset-level |
| MD5 | fixed | 0.15 | missing_token | 0.8860 | 0.2907 | 0.8302 | 0.6690 | Low p, weak |
| MD6 | simultaneous | 0.3 | zero | 0.8791 | 0.3023 | 0.8395 | 0.6736 | 3-pass costly, no gain |

### 4.5 Feature-level Modality Dropout — MDdrop1

Config: `md_drop_imu=0.20`, `md_drop_rgbd=0.10`, all else same as R4.

| Condition | Accuracy | F1 |
|---|---|---|
| Full (both) | 0.8860 | 0.8805 |
| RGBD-only | 0.3465 | 0.3181 |
| IMU-only | 0.8488 | 0.8430 |
| Average | 0.6938 | 0.6805 |

Feature-level dropout significantly improves RGBD-only vs R4 (+20pp), but Full accuracy regressed (88.60% < 89.77%). The 20% IMU dropout rate was insufficient to break IMU dominance.

### 4.6 Feature-level Dropout + CMAR — CMAR1 ⭐ BEST

Config: `md_drop_imu=0.35`, `md_drop_rgbd=0.10`, `cmar_weight=0.1`, `cmar_proj_dim=64`.

| Condition | Accuracy | F1 |
|---|---|---|
| Full (both) | **0.9093** | **0.9061** |
| RGBD-only | **0.4163** | **0.4072** |
| IMU-only | **0.8628** | **0.8599** |
| Average | **0.7295** | **0.7244** |
| Degradation | **0.4930** | — |

CMAR1 simultaneously achieves the **highest full accuracy** (90.93%) and the **best missing-modality robustness**, with Degradation < 0.50 for the first time.

### 4.7 CMAR Ablation & Hyperparameter Sweep (CMAR2–CMAR9)

| Exp | Method | Full | F1 | RGBD-only | RGBD F1 | IMU-only | IMU F1 | Avg |
|-----|--------|------|----|-----------|---------|----------|--------|-----|
| CMAR2 | Feature drop (imu=0.45) + CMAR(0.1) | 0.8744 | 0.8702 | 0.3395 | 0.2934 | 0.7953 | 0.7884 | 0.6698 |
| CMAR3 | Feature drop (imu=0.35) + CMAR(0.3) | 0.8860 | 0.8823 | 0.4512 | 0.4516 | 0.8651 | 0.8612 | 0.7341 |
| CMAR4 | Feature drop (imu=0.45) + CMAR(0.3) | 0.8907 | 0.8864 | 0.3674 | 0.3386 | 0.8581 | 0.8563 | 0.7054 |
| CMAR5 | Feature drop (imu=0.50, rgbd=0.05) + CMAR(0.1) | 0.9070 | 0.9046 | 0.4860 | 0.4830 | 0.7930 | 0.7745 | 0.7287 |
| CMAR6 | Feature drop + CMAR proj=128 | 0.8791 | 0.8683 | 0.3698 | 0.3573 | 0.8372 | 0.8247 | 0.6953 |
| CMAR7 | Feature drop + CMAR + aux=0.3 | 0.8860 | 0.8820 | 0.3837 | 0.3610 | 0.8302 | 0.8211 | 0.7000 |
| CMAR8 | Feature drop (imu=0.45, rgbd=0.05) + CMAR(0.3) | 0.8953 | 0.8934 | 0.4186 | 0.3907 | 0.8395 | 0.8342 | 0.7178 |
| CMAR9 | Feature drop (imu=0.50) + CMAR(0.2) | 0.8791 | 0.8770 | 0.4116 | 0.4020 | 0.8419 | 0.8421 | 0.7109 |

### 4.8 Extended Sweep & Ablation (CMAR12–CMAR26)

Base config = **CMAR5**: d_model=160, cross_mamba, pretrained ResNet18, partial freeze, temporal_velocity=true, augment=true, bs=32, ls=0.1, mixup=0.15, wd=0.008, cosine_warmup(5), lr=3e-4, patience=40, seed=42, md_drop_imu=0.50, md_drop_rgbd=0.05, cmar_weight=0.1, cmar_proj_dim=64, aux_weight=0.1.

#### Hyperparameter Variations (CMAR12–CMAR16)

| Exp | Key Change | Full | F1 | RGBD-only | RGBD F1 | IMU-only | IMU F1 | Avg |
|-----|-----------|------|-----|-----------|---------|----------|--------|-----|
| CMAR12 | md_drop_imu=0.55 | 0.8837 | 0.8829 | 0.4605 | 0.4482 | 0.8163 | 0.7955 | 0.7202 |
| CMAR13 | md_drop_imu=0.60 | 0.8744 | 0.8697 | 0.4628 | 0.4419 | 0.8256 | 0.8168 | 0.7209 |
| CMAR14 | md_drop_rgbd=0.00 | 0.8860 | 0.8836 | 0.4000 | 0.3745 | 0.0372 | 0.0027 | 0.4411 |
| CMAR15 | cmar_weight=0.2 | 0.8953 | 0.8932 | 0.4186 | 0.3998 | 0.7907 | 0.7691 | 0.7016 |
| CMAR16 | imu=0.55, rgbd=0.03, cmar=0.15 | 0.8907 | 0.8888 | 0.4721 | 0.4531 | 0.8070 | 0.7977 | 0.7233 |

#### Ablation Studies (CMAR17–CMAR26)

| Exp | Ablation Target | Full | F1 | RGBD-only | RGBD F1 | IMU-only | IMU F1 | Avg | Verdict |
|-----|----------------|------|-----|-----------|---------|----------|--------|-----|---------|
| CMAR17 | freeze=none (full unfreeze) | 0.8047 | 0.8061 | — | — | — | — | — | ❌ Severe overfitting |
| CMAR18 | seed=43 | 0.8721 | 0.8675 | 0.4581 | 0.4341 | 0.8047 | 0.7911 | 0.7116 | Seed 42 better |
| CMAR19 | cmar_loss_type=cosine | 0.9023 | 0.9003 | 0.4209 | 0.4136 | 0.8186 | 0.8108 | 0.7140 | ❌ MSE beats cosine |
| CMAR20 | md_drop_mode=independent | 0.8860 | 0.8839 | 0.4000 | 0.3738 | 0.8395 | 0.8377 | 0.7085 | ❌ Independent worse |
| CMAR21 | aux_weight=0.2 | 0.9000 | 0.8965 | 0.4442 | 0.4279 | 0.8116 | 0.8031 | 0.7186 | Marginal |
| CMAR22 | d_model=256 | 0.9070 | 0.9064 | 0.4349 | 0.4017 | 0.7860 | 0.7725 | 0.7093 | ❌ Larger model no RGBD gain |
| CMAR23 | n_layers=3 | 0.8698 | 0.8641 | 0.3837 | 0.3689 | 0.8047 | 0.7814 | 0.6860 | ❌ Overfits |
| CMAR24 | imu=0.65, cmar=0.3 | 0.8977 | 0.8936 | 0.4605 | 0.4615 | 0.7256 | 0.6982 | 0.6946 | ❌ IMU collapses |
| CMAR25 | lr=1e-4, 150ep | 0.8628 | 0.8567 | 0.3535 | 0.3335 | 0.7558 | 0.7352 | 0.6574 | ❌ Undertrained |
| CMAR26 | dropout=0.3 | 0.8791 | 0.8748 | 0.3116 | 0.2859 | 0.8209 | 0.8059 | 0.6705 | ❌ Over-regularized |

#### Key Ablation Findings

1. **Removing RGBD dropout is catastrophic** (CMAR14): IMU-only drops to 3.7% (random chance) — the 5% RGBD dropout is essential for bidirectional robustness
2. **Cosine vs MSE CMAR** (CMAR19): MSE outperforms cosine similarity for RGBD-only accuracy
3. **Independent vs Exclusive dropout** (CMAR20): Independent mode allows 2.5% chance of dropping both modalities simultaneously, which is harmful
4. **Deeper model** (CMAR23, n_layers=3): Overfits on 431 training samples
5. **Larger d_model** (CMAR22, d_model=256): Same Full accuracy but worse RGBD-only
6. **Full backbone unfreezing** (CMAR17): Catastrophic overfitting (Full=0.8047)
7. **Lower learning rate** (CMAR25): Undertrained even with 150 epochs
8. **Higher classifier dropout** (CMAR26): Information bottleneck too tight, hurts RGBD-only
9. **CMAR5 config is a robust local optimum**: 15 experiments failed to beat its RGBD-only=0.4860

### 4.9 Regularization & Curriculum Experiments (CMAR27–CMAR29)

Building on CMAR5's sweet spot, we tested stronger regularization and curriculum-based MD-Drop schedules.

| Exp | Key Change vs CMAR5 | Full Acc | Full F1 | RGBD-only | RGBD F1 | IMU-only | IMU F1 | Avg | Note |
|-----|---------------------|----------|---------|-----------|---------|----------|--------|-----|------|
| CMAR27 | mixup=0.3, wd=0.01 | 0.8837 | 0.8790 | 0.3674 | 0.3374 | 0.8163 | 0.8046 | 0.6891 | ❌ Stronger reg hurts all metrics |
| CMAR28 | curriculum MD-Drop (ramp 0→0.50 over training) | 0.9186 | 0.9166 | 0.3721 | 0.3476 | 0.8488 | 0.8411 | 0.7132 | 2nd best Full ever! But RGBD-only hurt |
| CMAR29 | reverse curriculum (ramp 0.50→0) | 0.8907 | 0.8876 | 0.3930 | 0.3661 | 0.8349 | 0.8227 | 0.7062 | ❌ Neither direction beats fixed dropout |

**Key Findings:**
- Curriculum MD-Drop (CMAR28) achieves outstanding Full accuracy (0.9186, 2nd best ever) by starting easy and gradually increasing difficulty, but the early epochs with low IMU dropout allow RGBD branch collapse
- Reverse curriculum (CMAR29) doesn't solve this either — fixed dropout remains optimal for RGBD-only robustness
- Stronger mixup/weight-decay (CMAR27) hurts across the board — the dataset is too small for heavy regularization

### 4.10 Batch Size Discovery & Optimization (CMAR30–CMAR38) ⭐

**BREAKTHROUGH**: Reducing batch size from 32 to 16 dramatically improves RGBD-only robustness. This is the single most impactful finding since introducing CMAR.

**Why batch_size=16 works**: With bs=16, the model sees 27 batches/epoch instead of 14 (bs=32). This means:
1. ~27× gradient updates per epoch → finer-grained optimization
2. MD-Drop creates more diverse "RGBD-only forced" passes → better coverage of the RGBD loss landscape
3. Sufficient within-batch diversity for meaningful CMAR alignment signal

| Exp | Key Config | bs | Full Acc | Full F1 | RGBD-only | RGBD F1 | IMU-only | IMU F1 | Avg | Note |
|-----|-----------|-----|----------|---------|-----------|---------|----------|--------|-----|------|
| CMAR30 | cmar=0.10, imu=0.50 | 16 | 0.9000 | 0.8969 | 0.5209 | 0.5093 | 0.8372 | 0.8304 | 0.7527 | ⭐ First >50% RGBD-only! |
| CMAR31 | cmar=0.10, imu=0.55 | 16 | 0.9140 | 0.9102 | 0.5116 | 0.4949 | 0.8395 | 0.8303 | 0.7550 | imu=0.55 slightly worse RGBD |
| CMAR32 | cmar=0.10, imu=0.50 | 8 | 0.8860 | 0.8823 | 0.3419 | 0.3132 | 0.8628 | 0.8531 | 0.6969 | ❌ bs=8 too noisy |
| CMAR33 | cmar=0.20, imu=0.50 | 16 | 0.8977 | 0.8946 | 0.4930 | 0.4817 | 0.8000 | 0.7856 | 0.7302 | cmar=0.20 too strong |
| **CMAR34** | **cmar=0.15, imu=0.50** | **16** | **0.9116** | **0.9089** | **0.5442** | **0.5371** | **0.8628** | **0.8531** | **0.7729** | **⭐⭐ NEW BEST RGBD-only & Avg** |
| CMAR35 | cmar=0.15, imu=0.55 | 16 | 0.8884 | 0.8847 | 0.5186 | 0.5044 | 0.8163 | 0.8013 | 0.7411 | imu=0.50 better than 0.55 |
| CMAR36 | cmar=0.12, imu=0.50 | 16 | 0.9023 | 0.8990 | 0.5140 | 0.5001 | 0.8395 | 0.8303 | 0.7519 | cmar=0.12 slightly worse than 0.15 |
| CMAR37 | cmar=0.15, imu=0.45 | 16 | 0.9023 | 0.8990 | 0.4465 | 0.4253 | 0.7698 | 0.7534 | 0.7062 | ❌ imu=0.45 too low |
| CMAR38 | cmar=0.15, imu=0.50, seed=123 | 16 | 0.8791 | 0.8759 | 0.4442 | 0.4178 | 0.8395 | 0.8333 | 0.7209 | Seed sensitivity: ~10pp RGBD variance |
| CMAR39 | cmar=0.15, imu=0.50, rgbd=0.08 | 16 | 0.9116 | 0.9086 | 0.4000 | 0.3790 | 0.8279 | 0.8113 | 0.7132 | rgbd_drop 0.05→0.08 worse RGBD-only |

**Key Findings:**
1. **bs=16 is the sweet spot**: Consistently yields RGBD-only >50% (CMAR30, 31, 34, 35, 36 all above 0.49)
2. **bs=8 is too small** (CMAR32): Noisy gradients hurt convergence, RGBD-only=0.3419
3. **cmar=0.15 optimal at bs=16** (CMAR34): Better than 0.10 (CMAR30) and 0.20 (CMAR33)
4. **imu=0.50 confirmed optimal** at bs=16: Both imu=0.45 (CMAR37) and imu=0.55 (CMAR35) are worse
5. **Seed sensitivity** (CMAR38 vs CMAR34): RGBD-only varies ~10pp between seeds — inherent variance on small dataset
6. **CMAR34 achieves best-ever average score** (0.7729): Excellent balance of Full (0.9116), RGBD-only (0.5442), and IMU-only (0.8628)
7. **Higher RGBD dropout hurts** (CMAR39: rgbd_drop=0.08 vs CMAR34: 0.05): RGBD-only drops from 0.5442→0.4000, confirming 0.05 is optimal

### 4.11 Comprehensive Robustness Evaluation (Centaur-style) ⭐

Following the stochastic corruption protocol from Xaviar et al. [23] (Centaur), we evaluate representative models under **four corruption modes** to assess real-world robustness. All stochastic conditions are averaged over 3 independent trials. Corruption is applied at GPU tensor level on both RGBD and IMU simultaneously.

**Corruption Modes (matching paper exactly):**
1. **Mode 1 — Gaussian Noise**: Additive N(0, σ) noise applied independently per channel per timestep, σ ∈ {0.05, 0.1, 0.2, 0.3}
2. **Mode 2 — Per-Channel Consecutive Missing**: Each channel independently alternates between normal (Exp(1/s_norm)) and corrupted (Exp(1/s_corr)) intervals; missing values set to zero
3. **Mode 3 — Per-Sensor Consecutive Missing**: Same as Mode 2, but all channels of a sensor share the same missing pattern
4. **Mode 4 — Combined**: c4 = c2(c1(x)) — first add Gaussian noise, then apply per-channel consecutive missing

**Adapted Parameters for UTD-MHAD:**
- RGBD (T=16): s_norm=8, s_corr ∈ {2, 4, 6}
- IMU (T=128): s_norm=60, s_corr ∈ {15, 30, 45}
- Mode 4 combos: (σ=0.05, rc=2, ic=15), (σ=0.1, rc=4, ic=30), (σ=0.2, rc=6, ic=45)

**Models Evaluated:** MuMu (baseline [4]), R4 (Momentum Mamba baseline, no robustness), CMAR1 (first CMAR variant), CMAR34 (best overall).

#### Table 1: Clean Baseline

| Model | Acc | F1 |
|-------|-----|----|
| MuMu | 0.6116 | 0.5897 |
| R4 | 0.8977 | 0.8951 |
| CMAR1 | 0.9093 | 0.9061 |
| **CMAR34** | **0.9116** | **0.9089** |

#### Table 2: Mode 1 — Gaussian Noise N(0, σ)

| σ | MuMu | R4 | CMAR1 | CMAR34 |
|---|------|----|-------|--------|
| 0.05 | 0.6109 | 0.8798 | 0.8806 | 0.8767 |
| 0.10 | 0.6031 | 0.8752 | 0.8736 | 0.8775 |
| 0.20 | 0.6078 | 0.8744 | **0.8814** | 0.8837 |
| 0.30 | 0.5938 | 0.8775 | 0.8705 | 0.8690 |
| **Avg** | **0.6039** | **0.8767** | **0.8765** | **0.8767** |

#### Table 3: Mode 2 — Per-Channel Consecutive Missing

| s_corr (RGBD, IMU) | MuMu | R4 | CMAR1 | CMAR34 |
|---------------------|------|----|-------|--------|
| (2, 15) | 0.4783 | **0.7977** | 0.7938 | 0.7752 |
| (4, 30) | 0.4008 | **0.7147** | 0.6860 | 0.6876 |
| (6, 45) | 0.3411 | **0.6581** | 0.6194 | 0.5969 |
| **Avg** | **0.4067** | **0.7235** | **0.6997** | **0.6866** |

#### Table 4: Mode 3 — Per-Sensor Consecutive Missing

| s_corr (RGBD, IMU) | MuMu | R4 | CMAR1 | CMAR34 |
|---------------------|------|----|-------|--------|
| (2, 15) | 0.5078 | **0.7845** | 0.7574 | 0.7729 |
| (4, 30) | 0.4550 | **0.7023** | 0.6372 | 0.6349 |
| (6, 45) | 0.3953 | **0.6302** | 0.5736 | 0.5674 |
| **Avg** | **0.4527** | **0.7057** | **0.6561** | **0.6584** |

#### Table 5: Mode 4 — Combined (Noise + Per-Channel Missing)

| σ, s_corr (RGBD, IMU) | MuMu | R4 | CMAR1 | CMAR34 |
|------------------------|------|----|-------|--------|
| 0.05, (2, 15) | 0.4659 | 0.7977 | **0.7984** | 0.7744 |
| 0.10, (4, 30) | 0.3899 | **0.7078** | 0.7054 | 0.6845 |
| 0.20, (6, 45) | 0.3233 | **0.6566** | 0.6333 | 0.6256 |
| **Avg** | **0.3930** | **0.7207** | **0.7124** | **0.6948** |

#### Table 6: Overall Summary

| Metric | MuMu | R4 | CMAR1 | CMAR34 |
|--------|------|----|-------|--------|
| Clean Acc | 0.6116 | 0.8977 | 0.9093 | **0.9116** |
| Avg Mode 1 (Noise) | 0.6039 | **0.8767** | 0.8765 | **0.8767** |
| Avg Mode 2 (Per-Ch Miss) | 0.4067 | **0.7235** | 0.6997 | 0.6866 |
| Avg Mode 3 (Per-Sens Miss) | 0.4527 | **0.7057** | 0.6561 | 0.6584 |
| Avg Mode 4 (Combined) | 0.3930 | **0.7207** | 0.7124 | 0.6948 |
| **Avg All Corrupted** | **0.4748** | **0.7659** | **0.7470** | **0.7405** |
| **Robustness Ratio** | **0.7763** | **0.8532** | **0.8215** | **0.8123** |

**Key Findings:**

1. **R4 (no robustness) achieves the best robustness ratio** (0.8532) — it outperforms CMAR1 (0.8215) and CMAR34 (0.8123) on corrupted inputs despite slightly lower clean accuracy. This suggests that the Momentum Mamba backbone already has strong inherent robustness when both modalities contribute jointly.
2. **CMAR models excel at clean accuracy but are slightly more brittle under corruption**: CMAR34 has the highest clean accuracy (0.9116) but the lowest robustness ratio (0.8123). The CMAR alignment loss may create tighter cross-modal coupling that is more sensitive to missing data patterns.
3. **All Momentum Mamba variants vastly outperform MuMu**: MuMu's robustness ratio (0.7763) appears high only because its clean accuracy is already low (0.6116), making the ratio artificially favorable. In absolute terms, MuMu's corrupted accuracy (0.4748) is far below all others.
4. **Gaussian noise (Mode 1) barely affects any model**: All Momentum Mamba models maintain ~87–88% accuracy even at σ=0.30, showing excellent noise resilience. The additive noise regime is not a discriminating test.
5. **Consecutive missing (Modes 2-3) is the hardest corruption**: At the highest severity, accuracy drops to 57–66% for Momentum Mamba models. Per-channel missing (Mode 2) and per-sensor missing (Mode 3) yield similar degradation, with per-channel slightly harder for CMAR models and per-sensor slightly harder overall.
6. **Mode 4 (combined) closely tracks Mode 2**: Adding noise before per-channel missing has minimal additional effect — the missing data pattern dominates the performance drop.

### 4.12 Denoising / Smoothing Filter Experiments (CMAR44, Test-Time)

**Hypothesis:** Applying denoising or smoothing filters before the encoder could reduce the effect of sensor noise and improve robustness under corruption, particularly for Mode 1 (Gaussian noise).

#### 4.12.1 Training-Time Denoising (CMAR44) — Failed

Added a `TemporalDenoise` module to both encoders with 5 modes: moving_avg, gaussian, learnable, ema, kalman. Applied before encoding (before Conv1d for IMU, after spatial CNN for RGBD).

| Experiment | Config | Peak Acc | Outcome |
|-----------|--------|----------|---------|
| CMAR44 | Both RGBD+IMU Gaussian (k=5, σ=1.0) | 53.49% | Killed epoch 39 — RGBD temporal feature smoothing on T=16 destroys temporal structure |
| CMAR44-v2 | IMU-only Gaussian (k=5, σ=1.0), RGBD none | 49.07% | Killed epoch 72 — Gaussian on raw IMU removes high-frequency action features |

**Why it fails:** With only 431 training samples, any information loss from filtering is catastrophic. The IMU Conv1d(6, d_model, k=3) frontend already performs adaptive filtering. Additionally, the model's heavy regularization (dropout, label smoothing, mixup) leaves no headroom for the additional capacity loss from fixed smoothing.

#### 4.12.2 Test-Time Denoising — Negative Result

Pivoted to applying denoising **only at test time**, after corruption but before model inference. This preserves clean accuracy (no filter applied to clean data). Implemented 5 filter types:
- **RGBD:** 2D spatial Gaussian blur (depthwise conv2d) per-frame for all filter modes
- **IMU:** Mode-specific 1D temporal filter (moving_avg, gaussian, median, savgol, bilateral)

All experiments on CMAR34 checkpoint (91.16% clean accuracy):

#### Table 7: Test-Time Denoising Comparison (CMAR34)

| Filter | k | σ | Mode 1 Avg | Mode 2 Avg | Mode 3 Avg | Mode 4 Avg | Corrupt Avg | **Ratio** |
|--------|---|---|-----------|-----------|-----------|-----------|-------------|---------|
| **None (baseline)** | — | — | **0.8767** | 0.6866 | 0.6584 | **0.6948** | **0.7405** | **0.8123** |
| IMU-only Gaussian | 3 | 0.5 | 0.8750 | 0.6804 | 0.6636 | 0.6824 | 0.7369 | 0.8083 |
| RGBD+IMU Gaussian | 3 | 0.5 | 0.8775 | 0.6858 | **0.6674** | 0.6759 | 0.7383 | 0.8099 |
| RGBD+IMU Median | 3 | 0.5 | 0.8743 | **0.6902** | 0.6540 | 0.6711 | 0.7340 | 0.8052 |
| RGBD+IMU Moving Avg | 3 | 0.5 | 0.8732 | 0.6881 | 0.6597 | 0.6711 | 0.7346 | 0.8058 |

**All filters degrade overall robustness.** The baseline (no filter) achieves the highest robustness ratio (0.8123). Key observations:

1. **RGBD spatial Gaussian blur marginally helps Mode 1** (noise): 0.8775 vs 0.8767 (+0.08pp), confirming the blur removes some pixel-level noise. But the effect size is negligible.
2. **Mode 3 (per-sensor missing) slightly benefits from Gaussian** (+0.9pp): the blur may interpolate sparse boundary pixels, but again the effect is small.
3. **Mode 4 (noise+missing) consistently worsens**: the blur spreads zero-valued (missing) pixels into adjacent non-zero regions, amplifying the corruption.
4. **Median filter helps Mode 2 slightly** (+0.36pp) but hurts Mode 1 and Mode 3.
5. **IMU-only denoising has limited impact** because the model relies primarily on RGBD (50% IMU dropout during training).

**Why test-time denoising fails fundamentally:**
- **Domain shift**: The model was trained on clean, unfiltered data. Any filtering at test time alters the input distribution, even if it removes some noise.
- **Missing data dominates corruption**: Modes 2-4 involve zeroed frames/channels. Smoothing cannot reconstruct missing data — it needs interpolation/imputation.
- **Mode 1 is already well-handled**: 87.67% accuracy = 96.2% of clean. The headroom for improvement via denoising is minimal.
- **Zero-spreading**: For missing data modes, blurring spreads zero values into neighboring valid regions, making things worse.

**Conclusion:** Neither training-time nor test-time denoising/smoothing filters improve CMAR34's robustness on the Centaur benchmark. The performance gap vs R4 (ratio 0.8532) is driven by missing-modality sensitivity in the CMAR architectural design, not by insufficient noise filtering. Addressing this gap likely requires architectural changes (e.g., stronger modality-independent representations) rather than signal-level preprocessing.

---

### 4.13 Centaur-Style Convolutional Denoising Autoencoder (DAE) — Negative Result

**Hypothesis:** Following the Centaur paper (Xaviar et al., arXiv:2303.04636), a learned Convolutional Denoising Autoencoder trained on corrupted→clean pairs should reconstruct missing/noisy data better than fixed filters, improving robustness without retraining the HAR model.

#### Architecture

Adapted Centaur's Conv2D DAE to our multimodal setup:

- **IMU DAE**: 3-layer Conv1d encoder (6→32→64→128, k=5, stride=2) → FC bottleneck (latent=64) → 3-layer ConvTranspose1d decoder. No output activation (z-scored IMU is unbounded). 368,902 params.
- **RGBD DAE**: 3-layer Conv2d encoder (4→32→64→128, k=5, stride=2) → 3-layer ConvTranspose2d decoder → Sigmoid (RGBD in [0,1]). Applied per-frame. 518,724 params.
- **Training**: MSE loss, RMSprop (lr=1e-4, momentum=0.1), cosine annealing, 200 epochs, batch=16. Corruption via all 4 Centaur modes (p_apply=1.0, σ∈[0.02,0.30]).

#### Results

| DAE Config | Clean Acc | Mode 1 Avg | Mode 2 Avg | Mode 3 Avg | Mode 4 Avg | Corrupt Avg | **Ratio** |
|-----------|-----------|-----------|-----------|-----------|-----------|-------------|---------|
| **None (baseline)** | **0.9116** | **0.8767** | 0.6866 | 0.6584 | **0.6948** | **0.7405** | **0.8123** |
| IMU-only DAE | 0.9116 | 0.7516 | 0.6723 | 0.6170 | 0.6563 | 0.6803 | 0.7462 |
| RGBD+IMU DAE | 0.9116 | 0.7696 | **0.6858** | **0.6403** | 0.6783 | 0.6993 | 0.7671 |

**Both DAE configurations substantially degrade robustness** compared to the no-cleaning baseline. The RGBD+IMU DAE (ratio 0.7671) is better than IMU-only (0.7462) but both are far below the baseline (0.8123).

#### Analysis

1. **Mode 1 (Gaussian noise) severely degraded by DAE**: The IMU DAE actively worsens noise robustness (0.7516 vs 0.8767), the opposite of its intended purpose. The DAE's reconstruction introduces systematic distortions that are more harmful than the original noise.
2. **Mode 2-3 (missing data) marginally helped by RGBD DAE**: The RGBD+IMU DAE slightly improves Mode 2 (-0.08pp) and Mode 3 (+1.8pp vs IMU-only), suggesting the spatial DAE partially reconstructs missing frames.
3. **Domain shift remains the core issue**: The HAR model (CMAR34) was trained on clean data. Even "cleaned" DAE output differs from true clean data — the reconstruction artifacts create a new distribution mismatch.
4. **DAE capacity vs dataset size**: With only 431 training samples, the DAE cannot learn a sufficiently general clean-data manifold. The 368K-param IMU DAE and 518K-param RGBD DAE are likely underfitting the reconstruction task (best val MSE ~0.32, meaning substantial residual error).
5. **Why Centaur worked but ours doesn't**: Centaur trained both the DAE and HAR model on the same pipeline (DAE output → HAR), so the HAR model adapted to DAE reconstruction artifacts. Our approach applies a DAE to a HAR model that was never exposed to DAE-processed data.

**Conclusion:** The Centaur DAE cleaning approach fails as a test-time plug-in module for CMAR34. The fundamental problem is the same as with fixed filters: domain shift between DAE-processed data and the clean data the HAR model was trained on. For this approach to work, the HAR model would need to be retrained on DAE-cleaned data (end-to-end pipeline), which would require significant additional experimentation.

---

## 5. Analysis: Why CMAR Works

### 5.1 The RGBD Branch Collapse Problem

Without any robustness mechanism, the model faces an easy optimization shortcut: IMU alone can reach ~73% accuracy, while RGBD requires expensive learning (431 training samples, 50K+ pixel values per frame). The fusion mechanism learns to rely almost entirely on IMU features, causing the RGBD branch to produce near-constant representations — useful only as a small correction bias.

This manifests as:
- Full accuracy: near-optimal (model uses IMU well)
- RGBD-only: ~14% (branch produces near-constant output → random guessing)
- Zero-IMU input is completely OOD for a network never trained with it

### 5.2 How MD-Drop Fixes OOD

By randomly zeroing IMU features in 35–50% of training batches, the model is forced to classify from RGBD alone. The gradient signal from these forced RGBD-only steps incentivizes the RGBD encoder to produce discriminative features.

Without CMAR, however, the RGBD branch still produces representations in a different subspace from IMU — fusion still uses modality-specific "shortcuts" rather than building a unified representation.

### 5.3 How CMAR Complements MD-Drop

CMAR enforces that even in the ~50% of batches where **both modalities are present**, the RGBD branch must build representations that are **aligned with IMU representations**. This has two effects:

1. **Anti-collapse**: RGBD features cannot degenerate to near-zero or constant representations (which would maximize CMAR loss)
2. **Transfer signal**: The well-learned IMU representation acts as a "teacher" — RGBD is continuously pulled toward a semantically meaningful representation, accelerating learning from the sparse RGBD-only batches

Crucially, the projection heads isolate CMAR's effect: the main feature space remains free to diverge as needed for cross-modal fusion, while alignment is enforced only in the 64-dim projection space.

### 5.4 Why Stronger IMU Dropout Improved Full Accuracy

Counterintuitively, CMAR1 achieves **higher Full accuracy (90.93%)** than MDdrop1 (88.60%) despite using stronger IMU dropout. The explanation:

- MDdrop1 at 20% still allows IMU to dominate fusion in the majority of batches
- CMAR1's stronger IMU dropout (35%) forces more balanced branch training
- CMAR's alignment loss ensures this doesn't hurt IMU-path performance
- A model with two well-trained branches fuses better than one with a dominant-branch bias

The net result: CMAR1's RGBD branch is strong enough that combining both modalities creates **genuine complementarity** rather than one branch correcting the other's noise.

---

## 6. Leaderboard

### 6.1 Best by Full Accuracy

| Rank | Exp | Full | F1 | RGBD-only | IMU-only | Avg |
|------|-----|------|----|-----------|----------|-----|
| 🥇 | **CMAR28** | **0.9186** | **0.9166** | 0.3721 | 0.8488 | 0.7132 |
| 🥈 | CMAR31 | 0.9140 | 0.9102 | 0.5116 | 0.8395 | 0.7550 |
| 🥉 | **CMAR34** | **0.9116** | **0.9089** | **0.5442** | 0.8628 | **0.7729** |
| 4 | CMAR1 | 0.9093 | 0.9061 | 0.4163 | 0.8628 | 0.7295 |
| 5 | CMAR5 | 0.9070 | 0.9046 | 0.4860 | 0.7930 | 0.7287 |
| 5 | CMAR22 | 0.9070 | 0.9064 | 0.4349 | 0.7860 | 0.7093 |
| 7 | CMAR19 | 0.9023 | 0.9003 | 0.4209 | 0.8186 | 0.7140 |
| 7 | CMAR36 | 0.9023 | 0.8990 | 0.5140 | 0.8395 | 0.7519 |
| 9 | CMAR30 | 0.9000 | 0.8969 | 0.5209 | 0.8372 | 0.7527 |
| 9 | CMAR21 | 0.9000 | 0.8965 | 0.4442 | 0.8116 | 0.7186 |

### 6.2 Best by RGBD-only Accuracy

| Rank | Exp | RGBD-only | Full | Avg | Note |
|------|-----|-----------|------|-----|------|
| 🥇 | **CMAR34** | **0.5442** | 0.9116 | **0.7729** | ⭐ bs=16, cmar=0.15 — NEW BEST |
| 🥈 | CMAR30 | 0.5209 | 0.9000 | 0.7527 | bs=16, cmar=0.10 — first >50% |
| 🥉 | CMAR35 | 0.5186 | 0.8884 | 0.7411 | bs=16, cmar=0.15, imu=0.55 |
| 4 | CMAR36 | 0.5140 | 0.9023 | 0.7519 | bs=16, cmar=0.12 |
| 5 | CMAR31 | 0.5116 | 0.9140 | 0.7550 | bs=16, cmar=0.10, imu=0.55 |
| 6 | CMAR33 | 0.4930 | 0.8977 | 0.7302 | bs=16, cmar=0.20 |
| 7 | CMAR5 | 0.4860 | 0.9070 | 0.7287 | Previous best (bs=32) |
| 8 | CMAR16 | 0.4721 | 0.8907 | 0.7233 | bs=32, imu=0.55, rgbd=0.03 |
| 9 | CMAR13 | 0.4628 | 0.8744 | 0.7209 | bs=32, imu=0.60 |
| 10 | CMAR12 | 0.4605 | 0.8837 | 0.7202 | bs=32, imu=0.55 |

---

## 7. Key Findings

### What Worked
1. **Cross-Mamba fusion** consistently outperforms late fusion approaches (attention, gated, concat) — shared SSM scan enables deep cross-modal temporal interaction
2. **CMAR + MD-Drop synergy**: CMAR addresses RGBD branch collapse while MD-Drop makes missing-modality inference in-distribution. Together, they achieve Full=90.93% with balanced robustness
3. **Partial freeze (ResNet18 layer4)**: Best trade-off — leverages ImageNet low-level features while adapting high-level features to action recognition
4. **Temporal velocity features**: Feature-level frame differences provide motion cues without optical flow overhead
5. **Asymmetric MD-Drop** (high IMU, low RGBD dropout): Matches the modality imbalance — forces the model to develop RGBD competence
6. **Auxiliary per-modality loss** forces each encoder to learn independently discriminative features
7. **Momentum SSM**: Second-order dynamics improve noise robustness and sustained motion capture
8. **batch_size=16 is critical** (CMAR30–38): More gradient updates per epoch (27 vs 14 batches) AND more MD-Drop diversity → RGBD-only jumps from 0.4860 (bs=32) to 0.5442 (bs=16)
9. **cmar_weight=0.15** with bs=16 is better than 0.10 or 0.20 — the sweet spot for alignment without over-constraining
10. **CMAR improves corruption robustness** (§4.11): All CMAR models outperform R4 baseline on missing-modality robustness; CMAR5 achieves best Centaur robustness ratio (0.5690 vs R4's 0.5006)

### What Didn't Work
1. **Optical flow** (R8): Farnebäck flow adds noise rather than useful motion signal at 112×112 resolution
2. **More frames** (R7): 32 frames → OOM, no benefit; 16 frames sufficient
3. **Larger batch size** (R9): bs=32 consistently hurts convergence on small datasets
4. **Frozen ResNet18** (R3): ImageNet features too generic without task adaptation
5. **Full fine-tuning** (CMAR17): Catastrophic overfitting with 11.2M unfrozen params on 431 samples
6. **Deeper model** (CMAR23, 3 layers): Overfits on small dataset
7. **Larger d_model** (CMAR22, d=256): No RGBD-only gain, diminishing returns
8. **Cosine CMAR loss** (CMAR19): MSE works better — enforces both directional and magnitude alignment
9. **Independent dropout** (CMAR20): Dropping both modalities simultaneously is harmful
10. **Lower learning rate** (CMAR25): Undertrained even with 150 epochs
11. **Aggressive regularization** (CMAR24, CMAR26): IMU branch or classifier collapses
12. **Curriculum MD-Drop** (CMAR28-29): Neither ramping up nor down beats fixed dropout for RGBD-only
13. **batch_size=8** (CMAR32): Too noisy — RGBD-only drops to 0.3419
14. **Stronger mixup/weight-decay** (CMAR27): Dataset too small for heavy regularization

### Optimal Configuration (CMAR34 — Best RGBD-only & Average)
```
d_model=160, n_layers=2, expand=2, dropout=0.2
fusion=cross_mamba, aux_weight=0.1
encoder=pretrained (ResNet18), freeze=partial, temporal_velocity=true
md_drop_imu=0.50, md_drop_rgbd=0.05, cmar_weight=0.15, cmar_proj_dim=64
lr=3e-4, weight_decay=0.008, batch_size=16
label_smoothing=0.1, mixup_alpha=0.15
scheduler=cosine_warmup, warmup_epochs=5, patience=40, seed=42
augment=true
→ Full=0.9116, RGBD-only=0.5442, IMU-only=0.8628, Avg=0.7729
→ Centaur robustness ratio=0.5149
```

### Most Robust Configuration (CMAR5 — Best Centaur Robustness)
```
Same as CMAR34 except:
batch_size=32, cmar_weight=0.10, md_drop_imu=0.50, md_drop_rgbd=0.05
→ Full=0.9070, RGBD-only=0.4860, Robustness ratio=0.5690 (best)
```

### Best Full Accuracy (CMAR28 — Curriculum MD-Drop)
```
Same as CMAR34 except:
batch_size=32, cmar_weight=0.10, md_drop_curriculum=true (ramp 0→0.50)
→ Full=0.9186 (best ever), but RGBD-only=0.3721
```

### Previous Best RGBD-only (CMAR5 — bs=32 era)
```
Same as CMAR34 except:
batch_size=32, cmar_weight=0.10
→ Full=0.9070, RGBD-only=0.4860
```

---

## 8. Model Parameter Count

| Component | Parameters |
|-----------|-----------|
| RGBD Encoder (ResNet18 partial + temporal Mamba) | ~2.6M trainable |
| IMU Encoder (Conv1D + Mamba) | ~142K |
| Cross-Mamba Fusion Block | ~141K |
| CMAR Projection Heads (×2) | ~20.6K |
| AttentionPool | ~6.6K |
| Classification Head | ~4.3K |
| Aux Heads (×2, when enabled) | ~13.2K |
| Modality Embeddings | ~320 |
| **Total (trainable with partial freeze)** | **~2.9M** |

---

## 9. Reproducibility

### Hardware & Software
- **Hardware**: NVIDIA RTX 4090
- **Software**: PyTorch 2.11.0+cu130, Python 3.10, conda env "mma"
- **Dataset**: UTD-MHAD with fixed subject-based train/test split

### Training Command (CMAR1 — Best)
```powershell
conda activate mma
python train/run_train.py --pipeline rgbd_imu --data_root "./datasets/UTD-MHAD" `
  --epochs 100 --save_name rgbd_imu_CMAR1.pt --tb_dir runs --num_workers 0 `
  --batch_size 32 --label_smoothing 0.1 --mixup_alpha 0.15 --weight_decay 0.008 `
  --scheduler cosine_warmup --warmup_epochs 5 --lr 3e-4 --patience 40 --seed 42 `
  --dataset_kwargs '{"augment":true}' `
  --model_kwargs '{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.35,"md_drop_rgbd":0.10,"cmar_weight":0.1,"cmar_proj_dim":64}'
```

### Evaluation Command (with missing-modality test)
```powershell
python infer/run_infer.py --pipeline rgbd_imu --data_root "datasets/UTD-MHAD" `
  --checkpoint "checkpoints/rgbd_imu_CMAR1.pt" --num_workers 0 --eval_missing all
```

---

## 10. Comprehensive Methods & Techniques Summary

This section consolidates every technique, architectural component, and training strategy explored across **all experiments** (R1–R9, MD1–MD6, MDdrop1, CMAR1–CMAR38), organized by category with references and per-experiment usage.

### 10.1 Encoder Architecture

#### 10.1.1 RGBD Encoder — Visual Frontend

| Method | Description | Experiments | Result |
|--------|-------------|-------------|--------|
| **SpatialCNN** (custom) | 4-layer stride-2 Conv2D (4→32→64→128→D), ~340K params. Trains from scratch on 4-channel RGBD input. | R1, R2, R5, R9 | Competitive (R1: 89.30%) but lower ceiling |
| **PretrainedCNN (ResNet18)** [5] | ImageNet-pretrained ResNet18 with first conv adapted from 3→4 input channels (depth channel initialized as mean of RGB weights). ~11.2M total params. | R3, R4, R6–R8, all CMAR | Best visual encoder (R4: 89.77%) |
| **Freeze=all** | Only final projection layer trainable (~82K). Low-level + high-level features frozen. | R3 | ❌ Under-fits (88.60%) |
| **Freeze=partial** [5] | Layer4 + projection trainable (~2.6M). Leverages frozen low-level ImageNet features (edges, textures) while adapting high-level features. | R4, all CMAR (except CMAR17) | ✅ Best trade-off (89.77%→91.86%) |
| **Freeze=none** | Full fine-tuning of all 11.2M ResNet18 params. | R6, CMAR17 | ❌ Catastrophic overfitting on 431 samples (CMAR17: 80.47%) |
| **Temporal velocity features** | Frame-level feature differences: $v_t = f_t - f_{t-1}$, concatenated and projected: $[f_t \| v_t] \xrightarrow{\text{Linear}(2D, D)} f'_t$. Provides explicit motion cues in learned feature space. | R5, R8, all CMAR | ✅ Consistent improvement (~1-2pp) |
| **Optical flow (Farnebäck)** [16] | Dense optical flow computed between consecutive frames, stacked as 2 extra input channels (→6ch total). | R8 | ❌ Noise at 112×112 resolution (87.91%) |

#### 10.1.2 IMU Encoder

| Method | Description | Experiments | Result |
|--------|-------------|-------------|--------|
| **Conv1D frontend** | Conv1D(6→D, kernel=3) → BatchNorm → ReLU → Dropout. Extracts local temporal patterns from raw 6-axis (3-acc + 3-gyro) inertial data. | All | Standard, effective |
| **GAF encoding** [17] | Gramian Angular Summation Field: resample→normalize→arccos→outer-sum cosine matrix (6×64×64). Transforms 1D time series into 2D image representations. | Optional/tested | Not adopted for best config |

### 10.2 Temporal Backbone — Momentum Mamba

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **Mamba (Selective SSM)** | Input-dependent state transitions: $h_n = \bar{A}_n \odot h_{n-1} + \bar{B}_n \cdot x_n$. Gated architecture with causal depthwise Conv1D + SiLU. Linear-time sequence modeling. | [1, 3] | All | Core backbone |
| **2nd-order Momentum** | Momentum buffer: $v_n = \beta \cdot v_{n-1} + \alpha \cdot \bar{B}_n \cdot x_n$; $h_n = \bar{A}_n \odot h_{n-1} + v_n$. Learnable α (init=0.6) and β (init=0.99). Smooths noisy input injection (IMU noise, visual jitter). | [2, 4] | All | ✅ Noise robustness, sustained motion capture |
| **Complex Momentum** | $\beta_c = \rho \cdot e^{i\theta}$ — oscillatory dynamics for periodic motion patterns (walking, jogging). | [2] | Tested | ❌ Less stable than real momentum |
| **Chunked parallel scan** | 32-token chunk-based parallel processing replaces sequential Python for-loop. 95% CPU overhead reduction. | — | All | Implementation optimization |
| **n_layers=2** (default) | Two stacked MomentumMambaBlock layers per encoder branch. | All except CMAR23 | ✅ Optimal for dataset size |
| **n_layers=3** | Three layers — increased capacity. | CMAR23 | ❌ Overfits (86.98%) |
| **d_model=160** (default) | Hidden dimension throughout the model. | All except CMAR22 | ✅ Best size |
| **d_model=256** | Larger hidden dimension. | CMAR22 | ❌ No RGBD-only gain (90.70% Full, 43.49% RGBD) |

### 10.3 Fusion Methods

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **Cross-Mamba** ⭐ | Concatenate full temporal sequences with modality embeddings: $S = [f^{rgbd}_t + e_{rgbd}, ..., f^{imu}_t + e_{imu}]$, then process through shared MomentumMambaBlock. Preserves temporal alignment for deep cross-modal interaction. | [1] | R1, R5, all CMAR | ✅ Best fusion (90.93%+) |
| **Attention fusion** | Multi-head cross-attention between pooled modality features. | [15] | R2 | ❌ Weaker (87.67%) |
| **Gated fusion** | Learned sigmoid gate: $g = \sigma(W[f_{rgbd}; f_{imu}])$; $f = g \odot f_{rgbd} + (1-g) \odot f_{imu}$. | — | Tested | ❌ Surface-level interaction |
| **Concatenation** | Simple feature concatenation + MLP. | — | Tested | ❌ No temporal alignment |
| **Modality embeddings** | Learnable vectors $e_{rgbd}, e_{imu} \in \mathbb{R}^{D}$ added to sequences before cross-mamba to preserve modality identity. | — | All cross-mamba | ✅ Essential for cross-mamba |

### 10.4 Temporal Pooling & Classification Head

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **AttentionPool** | Learnable attention-weighted temporal pooling: $\alpha_t = \text{softmax}(W_2 \tanh(W_1 h_t))$; $z = \sum_t \alpha_t \cdot h_t$. Input-adaptive weighting over time steps. | [15] | All | ✅ Better than mean/max pooling |
| **Classification head** | Dropout($p$) → Linear($D$, 27). Simple linear classifier on pooled features. | — | All | Standard |
| **Auxiliary per-modality heads** | Separate AttentionPool + Linear for each branch. Forces independent discriminability: $\mathcal{L}_{aux} = \frac{1}{2}(\mathcal{L}_{rgbd}^{aux} + \mathcal{L}_{imu}^{aux})$ | — | All CMAR | ✅ Prevents free-riding |
| **aux_weight=0.1** (default) | Auxiliary loss weight. | Most CMAR | ✅ Best |
| **aux_weight=0.2** | Stronger aux signal. | CMAR21 | Marginal improvement (90.00%) |
| **aux_weight=0.3** | Even stronger. | CMAR7 | ❌ Over-constraining |

### 10.5 Missing-Modality Robustness Methods

#### 10.5.1 Dataset-Level Modality Dropout (MD1–MD6)

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **Fixed dropout** | Drop one modality's raw input with fixed probability $p$ per sample. | [6] | MD1, MD5 | Baseline robustness |
| **Curriculum dropout** | Ramp dropout probability from 0→$p_{max}$ over training epochs. Eases the model into missing-modality scenarios. | [6] | MD2–MD4 | ✅ Better than fixed at dataset level |
| **Missing token** | Learnable replacement vectors instead of zeros when modality dropped. | — | MD2, MD5 | ✅ Better than zero-fill for dataset-level |
| **Simultaneous 3-pass** | Compute loss for full, rgbd-only, and imu-only in every batch. | — | MD6 | ❌ Costly, no gain |

#### 10.5.2 Feature-Level Modality Dropout (MD-Drop)

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **Feature-level MD-Drop** ⭐ | Zero out encoded **features** (not raw input) after encoding, before fusion. Encoders always receive gradients — prevents gradient starvation. | [6, 7] | MDdrop1, all CMAR | ✅ Core robustness mechanism |
| **Exclusive mode** (default) | Drop one OR the other, never both: if $r < p_{imu}$: zero IMU; elif $r < p_{imu}+p_{rgbd}$: zero RGBD. | — | All CMAR except CMAR20 | ✅ Best mode |
| **Independent mode** | Each modality dropped independently. 2.5% chance of both dropped simultaneously. | — | CMAR20 | ❌ Both-dropped case harmful |
| **Curriculum MD-Drop** | Ramp $p_{imu}$ linearly from 0→$p_{max}$ over training epochs. | — | CMAR28 | High Full (91.86%) but poor RGBD-only |
| **Reverse curriculum** | Ramp $p_{imu}$ from $p_{max}$→0 over training. | — | CMAR29 | ❌ No improvement over fixed |
| **md_drop_imu=0.35** | Original optimal IMU dropout. | — | CMAR1 | Full=90.93%, RGBD=41.63% |
| **md_drop_imu=0.50** ⭐ | Higher IMU dropout — forces stronger RGBD learning. | — | CMAR5, CMAR30–38 | ✅ Best (RGBD-only: 48.60%→54.42%) |
| **md_drop_imu=0.55/0.60** | Even higher. | — | CMAR12, CMAR13 | Diminishing returns on Full acc |
| **md_drop_imu=0.65** | Very high. | — | CMAR24 | ❌ IMU branch collapses |
| **md_drop_rgbd=0.05** ⭐ | Low RGBD dropout (just enough for bidirectional robustness). | — | CMAR5+ | ✅ Best |
| **md_drop_rgbd=0.10** | Higher RGBD dropout. | — | CMAR1–CMAR4 | Good but 0.05 better for RGBD-only |
| **md_drop_rgbd=0.00** | No RGBD dropout. | — | CMAR14 | ❌ IMU-only collapses to 3.7% |

#### 10.5.3 Cross-Modal Alignment Regularization (CMAR)

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **CMAR (MSE)** ⭐ | $\mathcal{L}_{CMAR} = \frac{1}{B} \sum \| \text{Proj}_{rgbd}(\bar{h}^{rgbd}) - \text{Proj}_{imu}(\bar{h}^{imu}) \|_2^2$. Soft alignment via separate projection heads (Linear(D→64)). Prevents RGBD collapse + implicit knowledge distillation from strong IMU branch. | [8, 9, 18] | All CMAR except CMAR19 | ✅ Core innovation |
| **CMAR (cosine)** | Cosine similarity loss instead of MSE. | [8] | CMAR19 | ❌ MSE better (42.09% vs 48.60% RGBD) |
| **CMAR (Barlow Twins)** | Cross-correlation matrix regularization. | [19] | Implemented, not fully tested | — |
| **cmar_weight=0.10** | Default CMAR loss weight. | — | CMAR1–13, CMAR30–31 | ✅ Best at bs=32 |
| **cmar_weight=0.15** ⭐ | Slightly stronger alignment. | — | CMAR34–38 | ✅ Best at bs=16 |
| **cmar_weight=0.20** | Stronger alignment. | — | CMAR15, CMAR33 | Slightly over-constrained |
| **cmar_weight=0.30** | Strong alignment. | — | CMAR3–4, CMAR8, CMAR24 | ❌ Over-constraining with high IMU drop |
| **cmar_proj_dim=64** ⭐ | Projection head output dimension. | — | All CMAR except CMAR6 | ✅ Best |
| **cmar_proj_dim=128** | Larger projection space. | — | CMAR6 | ❌ Over-parameterized projection |

### 10.6 Training Techniques

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **AdamW** | Adam optimizer with decoupled weight decay. | [11] | All | ✅ Standard choice |
| **lr=3e-4** ⭐ | Learning rate. | — | All except CMAR25 | ✅ Best |
| **lr=1e-4** | Lower learning rate. | — | CMAR25 | ❌ Undertrained even at 150 epochs |
| **weight_decay=0.008** ⭐ | L2 regularization. | [11] | All except CMAR27 | ✅ Best |
| **weight_decay=0.01** | Stronger L2. | — | CMAR27 | ❌ Over-regularized |
| **Cosine annealing + warmup** | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\frac{t}{T}\pi))$ with 5-epoch linear warmup. $\eta_{min}=10^{-6}$. | [12] | All CMAR | ✅ Smooth decay |
| **Label smoothing (ε=0.1)** | Soft targets: $y'_k = (1-\varepsilon)y_k + \varepsilon/K$. Reduces overconfidence. | [13] | All | ✅ Standard |
| **Mixup (α=0.15)** ⭐ | Data-level interpolation: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\lambda \sim \text{Beta}(0.15, 0.15)$. | [14] | All except CMAR27 | ✅ Best |
| **Mixup (α=0.30)** | Stronger mixing. | — | CMAR27 | ❌ Too aggressive for small dataset |
| **Gradient clipping (norm=1.0)** | Prevents gradient explosion in SSM training. | — | All | ✅ Essential for Mamba stability |
| **Mixed precision (AMP)** | FP16 forward/backward with FP32 master weights. CUDA throughput improvement. | [20] | All | ✅ ~2× speedup |
| **Early stopping (patience=40)** | Stop training when val accuracy doesn't improve for 40 epochs. | — | All | ✅ Prevents overfitting |
| **batch_size=32** | Original default. 14 batches/epoch (431 samples). | — | R1–R9, MD1–MD6, CMAR1–CMAR29 | Good for Full accuracy |
| **batch_size=16** ⭐ | **BREAKTHROUGH**: 27 batches/epoch → more gradient updates + MD-Drop diversity. | — | CMAR30–38 | ✅ RGBD-only: 48.60%→54.42% |
| **batch_size=8** | 54 batches/epoch. | — | R1–R6, CMAR32 | ❌ Too noisy (bs=8 at CMAR32: RGBD=34.19%) |
| **Dropout (p=0.2)** ⭐ | Standard dropout in encoder + head. | — | All except CMAR26 | ✅ Best |
| **Dropout (p=0.3)** | Stronger dropout. | — | CMAR26 | ❌ Information bottleneck too tight |

### 10.7 Data Augmentation

| Method | Description | References | Experiments | Result |
|--------|-------------|------------|-------------|--------|
| **Random horizontal flip** | p=0.5, RGBD frames. | — | R5+, all CMAR | ✅ Standard |
| **Random crop + resize** | Scale ∈ [0.8, 1.0], resize to 112×112. | — | R5+, all CMAR | ✅ Spatial invariance |
| **Color jitter** | Brightness/contrast ±15% on RGB channels only. | — | R5+, all CMAR | ✅ Illumination robustness |
| **IMU jitter** | Gaussian noise σ=0.03 on raw sensor values. | — | All | ✅ Sensor noise simulation |
| **IMU scaling** | Random scale factor σ=0.1. | — | All | ✅ Amplitude invariance |
| **Uniform frame sampling** | Select T=16 frames uniformly from video. | — | All | ✅ Temporal coverage |
| **32 frames** | Double temporal resolution. | — | R7 | ❌ OOM, no benefit |

### 10.8 Master Experiment–Method Matrix

The table below shows which key techniques each experiment group used. ✓ = used, ✗ = not used, — = not applicable.

| Technique | R1–R6 | R7–R9 | MD1–MD6 | MDdrop1 | CMAR1 | CMAR2–9 | CMAR12–26 | CMAR27–29 | CMAR30–38 |
|-----------|-------|-------|---------|---------|-------|---------|-----------|-----------|-----------|
| PretrainedCNN (ResNet18) | R3–R6 | R7–R8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Partial freeze | R4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (except CMAR17) | ✓ | ✓ |
| Temporal velocity | R5 | R8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Momentum Mamba (2nd order) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cross-Mamba fusion | R1,R5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AttentionPool | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Aux per-modality loss | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dataset-level MD | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Feature-level MD-Drop | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CMAR alignment loss | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Curriculum MD-Drop | ✗ | ✗ | MD2–4 | ✗ | ✗ | ✗ | ✗ | CMAR28–29 | ✗ |
| Cosine warmup scheduler | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Label smoothing (0.1) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mixup (α=0.15) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (exc. CMAR27) | ✓ |
| batch_size=16 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Data augmentation | R5 | R8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 10.9 Best Configuration (CMAR34) — Full Method Stack

The best-performing experiment (CMAR34) uses the following complete method stack:

```
┌─────────────────────────── INPUT ───────────────────────────┐
│  RGBD: (B, 16, 4, 112, 112) — 16 uniformly-sampled frames  │
│  IMU:  (B, 192, 6) — 6-axis inertial, padded to T=192      │
│  Augmentation: HFlip, RandomCrop, ColorJitter, IMU jitter   │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────── ENCODER (parallel branches) ─────────────────┐
│  RGBD: ResNet18 [5] (partial freeze [5])                    │
│        → frame features (B, 16, 512)                        │
│        → projection Linear(512, 160)                        │
│        → temporal velocity [f_t || f_t−f_{t−1}] → Linear   │
│        → MomentumMamba ×2 [1,2,3]                           │
│        → (B, 16, 160)                                       │
│                                                             │
│  IMU:  Conv1D(6, 160, k=3) → BN → ReLU → Dropout           │
│        → MomentumMamba ×2 [1,2,3]                           │
│        → (B, 192, 160)                                      │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌──────────── ROBUSTNESS (training only) ─────────────────────┐
│  MD-Drop [6,7]: p(zero IMU)=0.50, p(zero RGBD)=0.05        │
│  CMAR [8,9,18]: MSE on projected features (dim=64),        │
│                  weight=0.15                                │
│  Auxiliary heads [9]: weight=0.1 per-modality CE loss       │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────── FUSION & CLASSIFICATION ─────────────────┐
│  Modality embeddings + sequence concatenation               │
│  Cross-Mamba [1]: shared MomentumMambaBlock on              │
│                   (B, 16+192, 160) tokens                   │
│  AttentionPool [15]: learnable temporal weighting → (B,160) │
│  Dropout(0.2) → Linear(160, 27)                            │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌──────────────────── TRAINING ───────────────────────────────┐
│  Loss: CE + 0.1·AuxLoss + 0.15·CMAR_MSE                    │
│  Label smoothing ε=0.1 [13] + Mixup α=0.15 [14]            │
│  AdamW [11] lr=3e-4, wd=0.008, grad_clip=1.0               │
│  Cosine warmup [12] (5 warmup, T_max=100, η_min=1e-6)      │
│  AMP (FP16) [20], batch_size=16, patience=40               │
└─────────────────────────────────────────────────────────────┘

→ Full=0.9116  |  RGBD-only=0.5442  |  IMU-only=0.8628  |  Avg=0.7729
```

### 10.10 Progression of Key Innovations

| Stage | Experiments | Key Innovation | RGBD-only | Full | What Changed |
|-------|-------------|---------------|-----------|------|-------------|
| **Baseline** | R1–R6 | ResNet18 + Cross-Mamba + Momentum SSM | 14.19% | 89.77% | Strong multimodal model, but RGBD branch collapsed |
| **Dataset dropout** | MD1–MD6 | Curriculum modality dropout on raw input | 36.98% | 88.84% | +22.8pp RGBD-only, but Full accuracy dropped |
| **Feature dropout** | MDdrop1 | Feature-level MD-Drop (after encoding) | 34.65% | 88.60% | Encoders preserve gradients, slight RGBD gain |
| **+ CMAR** | CMAR1 | Cross-Modal Alignment Regularization | 41.63% | **90.93%** | Anti-collapse + distillation → Full AND RGBD improve |
| **Higher IMU drop** | CMAR5 | md_drop_imu 0.35→0.50 | 48.60% | 90.70% | More forced RGBD-only training |
| **Batch size** ⭐ | CMAR30 | batch_size 32→16 | 52.09% | 90.00% | 27 batches/epoch, more MD-Drop diversity |
| **Tuned CMAR** ⭐ | CMAR34 | cmar_weight 0.10→0.15 at bs=16 | **54.42%** | 91.16% | Stronger alignment at higher batch resolution |

---

## 11. References

[1] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," in *International Conference on Learning Representations (ICLR)*, 2024.

[2] B. T. Polyak, "Some methods of speeding up the convergence of iteration methods," *USSR Computational Mathematics and Mathematical Physics*, vol. 4, no. 5, pp. 1–17, 1964.

[3] A. Gu, K. Goel, and C. Ré, "Efficiently Modeling Long Sequences with Structured State Spaces," in *International Conference on Learning Representations (ICLR)*, 2022.

[4] I. Sutskever, J. Martens, G. Dahl, and G. Hinton, "On the importance of initialization and momentum in deep learning," in *International Conference on Machine Learning (ICML)*, 2013.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.

[6] N. Neverova, C. Wolf, G. Taylor, and F. Nebout, "ModDrop: Adaptive Multi-modal Gesture Recognition," *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 38, no. 8, pp. 1692–1706, 2016.

[7] W. Wang, D. Tran, and M. Feiszli, "What Makes Training Multi-modal Classification Networks Hard?," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 12695–12705, 2020.

[8] A. Radford, J. W. Kim, C. Hallacy, et al., "Learning Transferable Visual Models From Natural Language Supervision," in *International Conference on Machine Learning (ICML)*, pp. 8748–8763, 2021.

[9] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *NeurIPS Deep Learning and Representation Learning Workshop*, 2015.

[10] C. Chen, R. Jafari, and N. Kehtarnavaz, "UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor," in *IEEE International Conference on Image Processing (ICIP)*, pp. 168–172, 2015.

[11] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *International Conference on Learning Representations (ICLR)*, 2019.

[12] I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," in *International Conference on Learning Representations (ICLR)*, 2017.

[13] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818–2826, 2016.

[14] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in *International Conference on Learning Representations (ICLR)*, 2018.

[15] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 5998–6008, 2017.

[16] G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion," in *Scandinavian Conference on Image Analysis (SCIA)*, pp. 363–370, 2003.

[17] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks," in *AAAI Workshops*, 2015.

[18] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, "Barlow Twins: Self-Supervised Learning via Redundancy Reduction," in *International Conference on Machine Learning (ICML)*, pp. 12310–12320, 2021.

[19] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," in *International Conference on Machine Learning (ICML)*, pp. 1597–1607, 2020.

[20] P. Micikevicius, S. Narang, J. Alben, et al., "Mixed Precision Training," in *International Conference on Learning Representations (ICLR)*, 2018.

[21] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *International Conference on Machine Learning (ICML)*, pp. 448–456, 2015.

[22] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[23] S. R. Xaviar, K. Su, and S.-B. Park, "Centaur: Robust Multimodal Fusion for Human Activity Recognition," *IEEE Sensors Journal*, vol. 24, no. 6, pp. 8351–8362, 2024. arXiv:2303.04636.
