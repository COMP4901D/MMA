# COMP 4901D MMA — Repository Guide
Last Updated: 2026-03-31

## Project Overview

**MMA (Momentum Mamba / Momentum Multimodal Attention)** 是一个面向人体动作识别 (HAR) 的多模态深度学习项目。基于 UTD-MHAD 数据集，实现了三种不同模态组合的训练管线，并包含一个协作式多任务多模态融合基线 (MuMu)。

**核心方法:**
- **Momentum Mamba (MMA):** 在 Mamba 选择性状态空间模型 (SSM) [1] 中引入二阶动量机制 (real / complex)，增强时序建模能力
- **Multimodal MMA (RGB-D + IMU):** 基于跨模态 Mamba (Cross-Mamba) 架构，融合 RGB-D 视频和 IMU 传感器数据，支持 PretrainedCNN (ResNet18) [2] 和 SpatialCNN 编码器；引入 Feature-level Modality Dropout (MD-Drop) [3] 和 Cross-Modal Alignment Regularization (CMAR) 以提升单模态缺失鲁棒性 (**最佳: Full=0.9093, RGBD-only=0.4163, IMU-only=0.8628**)

**关键特性:**
- **统一训练 / 推理入口:** `train/run_train.py` + `infer/run_infer.py`，通过 Pipeline Registry 管理所有模型 (含 baseline)
- 主要训练管线：RGB-D + IMU (支持多种编码器与融合策略)
- Momentum SSM：二阶动量扫描 (real / complex 两种模式)
- **ConvNeXt-V2 预训练骨干网络**：可作为视觉编码器替换自定义 CNN，同时支持 IMU-GAF 图像编码
- 多种融合策略：attention、gated、concat
- 数据增强：Mixup、Random Erasing、水平翻转、亮度/对比度扰动
- 标准划分：奇数 subject 训练 (1,3,5,7)，偶数 subject 测试 (2,4,6,8)
- 早停、多种调度器 (Cosine / Warmup+Cosine / StepLR)、Label Smoothing、梯度裁剪
- **TensorBoard** 实时训练监控
- **可视化工具包 `vis/`**：训练曲线、数据检查、评估图表、模型分析

---

## Directory Structure

```
COMP 4901D/
├── train/
│   └── run_train.py                   # 统一训练入口 (Pipeline Registry + TensorBoard)
├── infer/
│   └── run_infer.py                   # 统一推理入口 (自动从 checkpoint 恢复配置)
├── script/
│   ├── run_train.ps1                  # PowerShell 训练启动器
│   └── run_infer.ps1                  # PowerShell 推理启动器
│
├── model/                             # 模型包 (所有网络组件)
│   ├── __init__.py
│   ├── layers.py                      # RMSNorm
│   ├── mamba.py                       # MomentumSSM + MomentumMambaBlock
│   ├── fusion.py                      # 7 种融合策略
│   ├── encoders.py                    # RGBDEncoder, IMUEncoder
│   ├── mma_utdmad.py                  # MomentumMambaHAR (纯 IMU)
│   ├── mma_mmtsa.py                   # MMA_MMTSA (Depth + IMU/GAF)
│   ├── mma_rgbd_imu.py                # MultimodalMMA (RGB-D + IMU) ★ Best
│   └── backbones/
│       ├── __init__.py
│       ├── convnextv2.py              # ConvNeXt-V2 预训练编码器 (via timm)
│       ├── pretrained_cnn.py          # PretrainedCNN (ResNet18, 4ch RGBD)
│       ├── light_cnn.py               # ResBlock + LightCNN + SE 注意力
│       └── spatial_cnn.py             # SpatialCNN (RGBD 空间特征)
│
├── datasets/                          # 数据集包
│   ├── __init__.py                    # ACTION_NAMES 常量 + 统一导出
│   ├── transforms.py                  # GAFEncoder (IMU → GASF 图像)
│   ├── utd_inertial.py                # UTDMADInertialDataset (纯 IMU)
│   ├── utd_depth_imu.py               # UTD_MHAD_Dataset (Depth + IMU/GAF)
│   ├── utd_rgbd_imu.py                # UTDMADRGBDIMUDataset (RGB-D + IMU)
│   └── UTD-MHAD/                      # 原始数据文件
│       ├── Depth/                     # 深度图 (.mat)
│       ├── Inertial/                  # 6轴 IMU (.mat)
│       ├── RGB-part1/ ~ RGB-part4/    # RGB 视频 (.avi)
│       ├── Skeleton/                  # 骨骼数据 (.mat)
│       └── Sample_Code/              # 官方示例代码
│
├── baselines/
│   └── MuMu/
│       └── MuMu.py                    # MuMu: 协作式多任务多模态融合基线
│
├── vis/                               # 可视化工具包
│   ├── __init__.py
│   ├── training.py                    # TrainingLogger, plot_training_curves, plot_lr_schedule
│   ├── data_inspection.py            # IMU/GAF/Depth/RGBD 数据检查
│   ├── evaluation.py                  # 混淆矩阵、per-class 指标、预测置信度
│   └── model.py                       # 梯度流、权重直方图、t-SNE、注意力可视化
│
├── checkpoints/                       # 已保存模型权重
│   ├── mma_mmtsa_best.pt
│   ├── mma_rgbd_imu_attention_real_best.pt
│   ├── mma_rgbd_imu_concat_real_best.pt
│   └── mma_rgbd_imu_gated_real_best.pt
│
├── mma_utdmad.py                      # 遗留入口 1: 纯 IMU (独立训练循环)
├── mma_mmast.py                       # 遗留入口 2: Depth + IMU (独立训练循环)
├── mma_rgbd_imu.py                    # 遗留入口 3: RGB-D + IMU (独立训练循环)
├── diagnose_dataset.py                # 数据集目录诊断工具
├── REPO_GUIDE.md                      # 本文件
├── requirements.txt
├── environment.yml
├── mac_environment.yml
└── mac_requirements.txt
```

---

## Unified Runner Architecture

### Pipeline Registry

`train/run_train.py` 和 `infer/run_infer.py` 共享同一 Pipeline 注册表，定义了每个管线的模型类、数据集类、IO 模式和默认参数：

| Pipeline  | Model Class         | Dataset Class           | input_mode | output_mode  |
|-----------|---------------------|-------------------------|------------|--------------|
| `rgbd_imu`| `MultimodalMMA`     | `UTDMADRGBDIMUDataset`  | unpack     | logits       |
| `utdmad`  | `MomentumMambaHAR`  | `UTDMADInertialDataset` | unpack     | logits       |
| `mmtsa`   | `MMA_MMTSA`         | `UTD_MHAD_Dataset`      | unpack     | tuple_first  |
| `mumu`    | `MuMu`              | (user-provided)         | list       | mumu         |

**IO 模式说明:**
- `input_mode`: `unpack` = `model(*inputs)`, `list` = `model(inputs)` — 适配不同模型的前向接口
- `output_mode`: `logits` = 直接返回 logits, `tuple_first` = `output[0]`, `mumu` = `output[1]` (y_target)

**自定义模型 (不在注册表中):**
通过 `--model_module` / `--model_class` / `--dataset_module` / `--dataset_class` 可训练任意模型，无需修改代码。

### 训练器 `train/run_train.py`

统一训练入口，支持：
- Pipeline 注册表 + 自定义模型/数据集
- JSON 格式 model_kwargs 和 dataset_kwargs
- 自动训练/测试集划分 (奇偶 subject)
- 归一化统计自动从 train → test 传播
- 多种优化器 (Adam / AdamW) 和调度器 (Cosine / Warmup+Cosine / StepLR)
- Mixup、Label Smoothing、梯度裁剪
- 早停 (基于 acc 或 f1)
- 最佳模型保存 (含完整恢复信息)
- **TensorBoard** 日志 (loss, acc, f1, lr, hparams)
- `vis` 模块可视化 (训练曲线、混淆矩阵、per-class 指标)

### 推理器 `infer/run_infer.py`

统一推理入口：
- 自动从 checkpoint 恢复 pipeline / model_kwargs (可通过 CLI 覆盖)
- 输出 classification_report + `.npz` 预测文件
- 可选 `--vis_dir` 生成混淆矩阵和 per-class 指标图

---

## Quick Start

### PowerShell 脚本

```powershell
# 训练 MMTSA (默认设置)
.\script\run_train.ps1 -Pipeline mmtsa -DataRoot "./datasets/UTD-MHAD"

# 训练 MMTSA，gated fusion，200 epochs，开启 TensorBoard
.\script\run_train.ps1 -Pipeline mmtsa -ModelKwargs '{"fusion":"gated"}' `
    -Epochs 200 -TbDir runs -VisDir vis_results

# 训练 RGBD-IMU，attention fusion + ConvNeXtV2 编码器
.\script\run_train.ps1 -Pipeline rgbd_imu `
    -ModelKwargs '{"fusion":"attention","encoder":"convnextv2"}' `
    -DataRoot "./datasets/UTD-MHAD"

# 训练 UTD-MAD 纯 IMU
.\script\run_train.ps1 -Pipeline utdmad -DataRoot "./datasets/UTD-MHAD/Inertial"

# 训练自定义模型 (不在注册表中)
.\script\run_train.ps1 -Pipeline "" `
    -ModelModule "baselines.MuMu.MuMu" -ModelClass "MuMu" `
    -DatasetModule "datasets.utd_inertial" -DatasetClass "UTDMADInertialDataset" `
    -DataRoot "./datasets/UTD-MHAD/Inertial" -InputMode list -OutputMode mumu

# 推理
.\script\run_infer.ps1 -Pipeline mmtsa -Checkpoint checkpoints/mma_mmtsa_best.pt `
    -VisDir vis_results
```

### 直接 Python 调用

```bash
# 训练
python train/run_train.py --pipeline utdmad --data_root datasets/UTD-MHAD/Inertial
python train/run_train.py --pipeline mmtsa  --data_root datasets/UTD-MHAD \
       --model_kwargs '{"fusion":"gated"}' --tb_dir runs

# 推理
python infer/run_infer.py --pipeline mmtsa --data_root datasets/UTD-MHAD \
       --checkpoint checkpoints/mma_mmtsa_best.pt --vis_dir vis_results
```

### TensorBoard

```bash
tensorboard --logdir runs
```

训练时指定 `--tb_dir runs` (或 PS1 中 `-TbDir runs`)，TensorBoard 将记录：
- `Loss/train`, `Loss/val` — 每 epoch 训练/验证损失
- `Accuracy/train`, `Accuracy/val` — 每 epoch 准确率
- `F1/val` — 验证集加权 F1
- `LR` — 当前学习率
- **HParams** — 训练结束后记录超参数与最终指标的关联

---

## Unified Runner CLI Reference

### `train/run_train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pipeline` | (none) | 注册管线: `utdmad` / `mmtsa` / `rgbd_imu` / `mumu` |
| `--model_module` | (auto) | 自定义模型的 Python 模块路径 |
| `--model_class` | (auto) | 模型类名 |
| `--dataset_module` | (auto) | 数据集模块路径 |
| `--dataset_class` | (auto) | 数据集类名 |
| `--model_kwargs` | `""` | JSON 格式模型构造参数，如 `'{"fusion":"gated"}'` |
| `--dataset_kwargs` | `""` | JSON 格式数据集构造参数 |
| `--data_root` | (required) | 数据集根目录 |
| `--input_mode` | (auto) | `unpack` / `list` |
| `--output_mode` | (auto) | `logits` / `tuple_first` / `mumu` |
| `--epochs` | 100 | 最大训练轮数 |
| `--batch_size` | 16 | 批大小 |
| `--lr` | 5e-4 | 学习率 |
| `--weight_decay` | 1e-3 | 权重衰减 |
| `--optimizer` | `adamw` | `adam` / `adamw` |
| `--scheduler` | `cosine` | `cosine` / `cosine_warmup` / `step` / `none` |
| `--warmup_epochs` | 10 | Warmup 轮数 (仅 cosine_warmup) |
| `--step_size` | 30 | StepLR 步长 |
| `--step_gamma` | 0.1 | StepLR 衰减系数 |
| `--clip_grad` | 1.0 | 梯度裁剪范数 (0 = 禁用) |
| `--label_smoothing` | 0.0 | 标签平滑 |
| `--mixup_alpha` | 0.0 | Mixup α (0 = 禁用) |
| `--patience` | 20 | 早停耐心值 (0 = 禁用) |
| `--early_stop_metric` | `acc` | 早停指标: `acc` / `f1` |
| `--save_dir` | `checkpoints` | 模型保存目录 |
| `--save_name` | `best.pt` | 保存文件名 |
| `--resume` | (none) | 恢复训练的 checkpoint 路径 |
| `--seed` | 42 | 随机种子 |
| `--vis_dir` | `""` | 可视化输出目录 (空 = 禁用) |
| `--tb_dir` | `""` | TensorBoard 日志目录 (空 = 禁用) |
| `--device` | auto | `cuda` / `cpu` |
| `--num_workers` | 4 | DataLoader worker 数 |

### `infer/run_infer.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pipeline` | (auto from ckpt) | 管线名 |
| `--checkpoint` | (required) | Checkpoint 路径 |
| `--data_root` | (required) | 数据集根目录 |
| `--model_kwargs` | (auto from ckpt) | 覆盖模型参数 |
| `--dataset_kwargs` | `""` | 数据集参数 |
| `--output` | `preds.npz` | 预测输出文件 |
| `--vis_dir` | `""` | 可视化输出目录 |
| `--batch_size` | 32 | 批大小 |

### `script/run_train.ps1`

全部 Python 参数均映射为 PowerShell 命名参数，额外支持：
- `-CUDA`: 设置 `CUDA_VISIBLE_DEVICES`
- `-TbDir`: TensorBoard 日志目录

---

## Module Details

### 模型包 `model/`

#### `model/layers.py` — 共享基础层
- `RMSNorm`: Root Mean Square 归一化

#### `model/mamba.py` — Momentum Mamba 核心
- `MomentumSSM`: 二阶动量选择性状态空间扫描 (vanilla / real / complex)
- `MomentumMambaBlock`: RMSNorm → 双分支线性投影 → 因果深度卷积 → SSM → 门控 + 残差

#### `model/fusion.py` — 融合策略 (7 种)
- **MMTSA 族** (返回 `(fused, alpha)` 元组):
  `MomentumAttentionFusion`, `AttentionFusionNoMomentum`, `SegmentAttention`, `GatedFusion`, `ConcatFusion`
- **RGBD-IMU 族** (返回单个张量):
  `CrossModalAttentionFusion`, `GatedFusionSimple`

#### `model/backbones/` — 骨干网络子包
- `ConvNeXtV2Encoder`: timm 预训练编码器，支持任意输入通道数、部分冻结
- `PretrainedCNN`: ResNet18 预训练编码器，4 通道 RGBD 输入，支持 freeze="all"/"partial"/"none"
- `LightCNN`: 4 层 Conv2D + SE 注意力
- `SpatialCNN`: 4 层 stride-2 Conv2D (RGBD 帧)

#### `model/encoders.py` — 复合编码器
- `RGBDEncoder`: SpatialCNN/PretrainedCNN/ConvNeXtV2 (per-frame) → [temporal_velocity] → MomentumMamba (temporal)
- `IMUEncoder`: Conv1D/ConvNeXtV2-GAF → MomentumMamba (temporal)

#### 模型类汇总

| 类名 | 参数量 | 输入 | 输出 |
|------|--------|------|------|
| `MomentumMambaHAR` | ~321K | `(B, L, 6)` | `(B, 27)` logits |
| `MMA_MMTSA` | ~4.3M | `(depth, imu)` | `(logits, aux_dict)` |
| `MultimodalMMA` | ~2.9M (trainable) | `(rgbd, imu)` | `(B, 27)` logits |
| `MuMu` | ~448K | `[x_list]` | `(y_aux, y_target, alpha, attn)` |

### 数据集包 `datasets/`

| 类名 | 模态 | 返回 | norm_keys |
|------|------|------|-----------|
| `UTDMADInertialDataset` | 纯 IMU | `(x, y)` | `mean`, `std` |
| `UTD_MHAD_Dataset` | Depth + IMU/GAF | `(depth, imu, label)` | — |
| `UTDMADRGBDIMUDataset` | RGB-D + IMU | `(rgbd, imu, label)` | `iner_mean`, `iner_std` |

### 可视化工具包 `vis/`

| 模块 | 导出函数 |
|------|----------|
| `vis.training` | `TrainingLogger`, `plot_training_curves`, `plot_lr_schedule` |
| `vis.data_inspection` | `plot_imu_signal`, `plot_imu_comparison`, `plot_gaf_image`, `plot_gaf_segments`, `plot_depth_frames`, `plot_depth_representation`, `plot_rgbd_frames`, `plot_batch_overview`, `plot_data_distribution`, `plot_class_distribution` |
| `vis.evaluation` | `plot_confusion_matrix`, `plot_confusion_matrix_from_array`, `plot_per_class_metrics`, `plot_per_class_accuracy`, `plot_prediction_confidence` |
| `vis.model` | `plot_gradient_flow`, `plot_param_histogram`, `plot_feature_tsne`, `plot_attention_weights`, `plot_fusion_alpha`, `plot_model_size` |

### 基线 `baselines/MuMu/MuMu.py`

MuMu 协作式多任务多模态融合：
- `UnimodalFeatureEncoder` → `SMFusion` → `GMFusion` → 双分类头
- `MuMuLoss`: 加权联合损失 `L_target + β · L_aux`
- 已注册为 `mumu` 管线，可直接通过统一训练器运行

---

## Core Algorithms

### Momentum Selective State-Space Model (MomentumSSM)

| 模式 | 递推公式 | 特点 |
|------|----------|------|
| **Vanilla** (`none`) | $h_n = \bar{A}_n \odot h_{n-1} + \bar{B}_n \cdot x_n$ | 一阶，标准 Mamba |
| **Real** (`real`) | $v_n = \beta \cdot v_{n-1} + \alpha \cdot \bar{B}_n \cdot x_n$; $h_n = \bar{A}_n \odot h_{n-1} + v_n$ | 二阶 EMA 平滑 |
| **Complex** (`complex`) | $\beta_c = \rho \cdot e^{i\theta}$ | 振荡记忆，捕获周期性 |

### Gramian Angular Field (GAF)

1D 时间序列 → 2D 图像: 归一化 → $\phi = \arccos(\text{scaled})$ → $G[i,j] = \cos(\phi_i + \phi_j)$ → 灰度

### Feature-level Modality Dropout (MD-Drop)

在 `MultimodalMMA.forward()` 编码之后、融合之前，以概率随机将某一模态特征全部置零：

```
r = random.random()
if r < md_drop_imu:   fi = zeros_like(fi)   # RGBD-only path
elif r < md_drop_imu + md_drop_rgbd:  fv = zeros_like(fv)   # IMU-only path
# else: both modalities present
```

这迫使模型在训练时见到缺失模态的输入，消除推理时 out-of-distribution 问题。

构造参数: `md_drop_imu` (default 0.0), `md_drop_rgbd` (default 0.0)。

### Cross-Modal Alignment Regularization (CMAR)

在编码之后对两个分支施加表示对齐约束，防止弱势模态（RGBD）彻底"摆烂"：

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_{aux} \cdot \mathcal{L}_{aux} + \lambda_{cmar} \cdot \| \text{Proj}_{rgbd}(\bar{h}^{rgbd}) - \text{Proj}_{imu}(\bar{h}^{imu}) \|^2$$

其中 $\bar{h}$ 为时间维度上的 mean-pool，两个 `Linear(d_model → cmar_proj_dim)` 投影到共享对齐空间。
即使 IMU 分支已能准确分类，CMAR 也会强制 RGBD 特征向 IMU 特征对齐，避免 RGBD 编码器退化为常量输出。

构造参数: `cmar_weight` (default 0.0), `cmar_proj_dim` (default 64)。
CMAR loss 由 `MultimodalMMA` 存储于 `_cmar_loss`，由 `train/default_train.py` 读取并加入总损失。

---

## Dependency Graph

```
model/layers.py          # RMSNorm (零依赖)
    ↑
model/mamba.py           # MomentumSSM, MomentumMambaBlock
    ↑
model/backbones/         # ConvNeXtV2, LightCNN, SpatialCNN (零依赖)
model/fusion.py          # 7 种融合器 (零依赖)
    ↑
model/encoders.py        # RGBDEncoder, IMUEncoder (← mamba, backbones)
    ↑
model/mma_utdmad.py      # MomentumMambaHAR (← mamba)
model/mma_mmtsa.py       # MMA_MMTSA (← fusion, backbones)
model/mma_rgbd_imu.py    # MultimodalMMA (← encoders, fusion)

datasets/transforms.py   # GAFEncoder (零依赖)
    ↑
datasets/utd_inertial.py
datasets/utd_depth_imu.py (← transforms)
datasets/utd_rgbd_imu.py  (← transforms)

train/run_train.py ──→ model.*, datasets.*, vis.training, vis.evaluation, tensorboard
infer/run_infer.py ──→ model.*, datasets.*, vis.evaluation
```

---

## Data Flow

### Pipeline 1: 纯 IMU (`utdmad`)
```
Inertial .mat (T,6) → pad/truncate → (256,6) → MomentumMambaHAR → logits (B,27)
```

### Pipeline 2: Depth + IMU/GAF (`mmtsa`)
```
Depth (T,H,W) → resize → 分段 → (N,3,64,64) ─┐
IMU   (T,6)   → GAF 编码 → (N,6,64,64) ───────┘ → MMA_MMTSA → logits (B,27)
```

### Pipeline 3: RGB-D + IMU (`rgbd_imu`)
```
RGB+Depth → RGBD (N,4,112,112) ─┐
IMU (T,6) → 归一化 ─────────────┘ → MultimodalMMA → logits (B,27)
```

**关键 model_kwargs 说明 (仅 rgbd_imu):**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `encoder` | str | `"spatial_cnn"` | `"pretrained"` = ResNet18, `"convnextv2"` = ConvNeXtV2 |
| `freeze` | str | `"all"` | `"partial"` = layer4+proj 可训练, `"none"` = 全量微调 |
| `temporal_velocity` | bool | `false` | 特征速度: $v_t = f_t - f_{t-1}$, 与 $f_t$ 拼接后投影 |
| `md_drop_imu` | float | `0.0` | Feature-level dropout: 以该概率将 IMU 特征归零 (训练期间) |
| `md_drop_rgbd` | float | `0.0` | Feature-level dropout: 以该概率将 RGBD 特征归零 (训练期间) |
| `cmar_weight` | float | `0.0` | CMAR 损失权重 $\lambda_{cmar}$；0 = 禁用 |
| `cmar_proj_dim` | int | `64` | CMAR 对齐投影维度 |
| `aux_weight` | float | `0.0` | 辅助分类头损失权重 |
| `md_schedule` | str | `"none"` | 数据集级 Modality Dropout: `"fixed"`, `"curriculum"`, `"simultaneous"` |

---

## Dataset: UTD-MHAD

- **规模:** 27 动作类别 × 8 subjects × 4 trials ≈ 861 样本
- **划分:** 奇数 subjects (1,3,5,7) → 训练 (431), 偶数 subjects (2,4,6,8) → 测试 (430)
- **文件命名:** `a{action}_s{subject}_t{trial}_{modality}.mat`

| ID | 动作 | ID | 动作 | ID | 动作 |
|----|------|----|------|----|------|
| 1 | swipe_left | 10 | draw_circle_CCW | 19 | knock |
| 2 | swipe_right | 11 | draw_triangle | 20 | catch |
| 3 | wave | 12 | bowling | 21 | pickup_throw |
| 4 | clap | 13 | boxing | 22 | jog |
| 5 | throw | 14 | baseball_swing | 23 | walk |
| 6 | arm_cross | 15 | tennis_swing | 24 | sit2stand |
| 7 | basketball_shoot | 16 | arm_curl | 25 | stand2sit |
| 8 | draw_x | 17 | tennis_serve | 26 | lunge |
| 9 | draw_circle_CW | 18 | push | 27 | squat |

---

## Saved Checkpoints

| 文件 | 管线 | 融合 | Acc | 说明 |
|------|------|------|-----|------|
| `checkpoints/rgbd_imu_CMAR1.pt` | rgbd_imu | cross_mamba | **0.9093** | ★ RGBD+IMU 最佳 (CMAR + MD-Drop) |
| `checkpoints/rgbd_imu_MDdrop1.pt` | rgbd_imu | cross_mamba | 0.8860 | MD-Drop 消融 |
| `checkpoints/rgbd_imu_R4.pt` | rgbd_imu | cross_mamba | 0.8977 | 无缺失鲁棒化基线 |
| `checkpoints/mma_mmtsa_best.pt` | mmtsa | attention | — | MMTSA 基线 |
| `checkpoints/mma_rgbd_imu_attention_real_best.pt` | rgbd_imu | attention | — | 早期实验 |
| `checkpoints/mma_rgbd_imu_concat_real_best.pt` | rgbd_imu | concat | — | 早期实验 |
| `checkpoints/mma_rgbd_imu_gated_real_best.pt` | rgbd_imu | gated | — | 早期实验 |

---

## Best Results

| Pipeline | Model | Best Acc | Best F1 | RGBD-only | IMU-only | Key Config | Checkpoint |
|----------|-------|----------|---------|-----------|----------|------------|------------|
| **RGBD + IMU (robust)** | MultimodalMMA | **0.9093** | **0.9061** | **0.4163** | **0.8628** | CMAR+MDdrop, md_drop_imu=0.35, cmar_weight=0.1 | `rgbd_imu_CMAR1.pt` |
| RGBD + IMU (baseline) | MultimodalMMA | 0.8977 | 0.8951 | 0.1419 | 0.7302 | ResNet18 partial, vel+aug, cross_mamba, aux=0.1 | `rgbd_imu_R4.pt` |

**推荐方法:**
- **需要缺失鲁棒性时:** RGBD+IMU CMAR1 — 90.93% 准确率，且在单模态缺失下保持鲁棒 (RGBD 41.63%, IMU 86.28%)
- **不需要缺失鲁棒性时:** RGBD+IMU R4 — 89.77%，更快训练 (bs=8)

详细实验报告见 `experiment_report.md`。

---

## Environment Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate mma
```

核心依赖: `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tensorboard`, `timm`, `opencv-python`

---

## References

[1] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," *ICLR*, 2024.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.

[3] N. Neverova, C. Wolf, G. Taylor, and F. Nebout, "ModDrop: Adaptive Multi-modal Gesture Recognition," *IEEE TPAMI*, 2016.
