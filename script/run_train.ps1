<# .SYNOPSIS
  Unified training launcher for all MMA HAR pipelines.

.DESCRIPTION
  Calls train/run_train.py with the chosen pipeline and parameters.
  Supports all registered pipelines (utdmad, mmtsa, rgbd_imu, mumu)
  as well as fully custom models via -ModelModule / -ModelClass.

.EXAMPLE
  # Train MMTSA with default settings
  .\script\run_train.ps1 -Pipeline mmtsa

  # Train MMTSA with gated fusion and more epochs
  .\script\run_train.ps1 -Pipeline mmtsa -ModelKwargs '{"fusion":"gated"}' -Epochs 200

  # Train RGBD-IMU with concat fusion
  .\script\run_train.ps1 -Pipeline rgbd_imu -ModelKwargs '{"fusion":"concat","imu_encoder":"convnextv2"}'

  # Train UTD-MAD inertial-only
  .\script\run_train.ps1 -Pipeline utdmad -DataRoot "./datasets/UTD-MHAD/Inertial"

  # Train a custom model not in the registry
  .\script\run_train.ps1 -ModelModule "baselines.MuMu.MuMu" -ModelClass "MuMu" `
      -DatasetModule "datasets.utd_inertial" -DatasetClass "UTDMADInertialDataset" `
      -DataRoot "./datasets/UTD-MHAD/Inertial" -InputMode list -OutputMode mumu

  # Train RGBD-IMU with consecutive modality dropout (sensor failure simulation)
  .\script\run_train.ps1 -Pipeline rgbd_imu -CorruptionMode consecutive `
      -CorruptionPConsecutive 0.4 -CorruptionBlockRange 3,8

  # Train RGBD-IMU with full modality dropout on IMU only
  .\script\run_train.ps1 -Pipeline rgbd_imu -CorruptionMode full `
      -CorruptionPFull 0.3 -CorruptionModalities "imu"

  # Train with mixed corruption (randomly picks full or consecutive per sample)
  .\script\run_train.ps1 -Pipeline rgbd_imu -CorruptionMode mixed `
      -CorruptionPFull 0.2 -CorruptionPConsecutive 0.3
#>

param(
    # ── Pipeline selection ──
    [string]$Pipeline = "mmtsa",          # utdmad | mmtsa | rgbd_imu | mumu | "" for custom

    # ── Custom model / dataset (override pipeline or use standalone) ──
    [string]$ModelModule = "",           # e.g. "model.mma_utdmad"
    [string]$ModelClass = "",           # e.g. "MomentumMambaHAR"
    [string]$DatasetModule = "",
    [string]$DatasetClass = "",

    # ── Kwargs (JSON strings) ──
    [string]$ModelKwargs = "",          # e.g. '{"fusion":"gated","feat_dim":256}'
    [string]$DatasetKwargs = "",          # e.g. '{"gaf_size":32}'

    # ── Data ──
    [string]$DataRoot = "./datasets/UTD-MHAD",

    # ── IO modes (auto if empty, override for custom models) ──
    [string]$InputMode = "",             # unpack | list
    [string]$OutputMode = "",             # logits | tuple_first | mumu

    # ── Training hyperparameters ──
    [int]$Epochs = 120,
    [int]$BatchSize = 16,
    [double]$LR = 5e-4,
    [double]$WeightDecay = 1e-3,
    [string]$Optimizer = "adamw",        # adam | adamw
    [string]$Scheduler = "cosine",       # cosine | cosine_warmup | step | none
    [int]$WarmupEpochs = 10,
    [double]$ClipGrad = 1.0,
    [double]$LabelSmoothing = 0.0,
    [double]$MixupAlpha = 0.0,
    [int]$Patience = 25,
    [string]$EarlyStopMetric = "acc",     # acc | f1
    [int]$Seed = 42,

    # ── Corruption / modality dropout ──
    [string]$CorruptionMode = "none",        # none | full | consecutive | mixed
    [double]$CorruptionPFull = 0.2,           # per-modality full-dropout probability
    [double]$CorruptionPConsecutive = 0.3,    # per-timestep block-start probability
    [int[]]$CorruptionBlockRange = @(2, 6),    # (min, max) consecutive block length
    [string[]]$CorruptionModalities = @("rgbd", "imu"),  # eligible modalities
    [double]$CorruptionBothDropProb = 0.0,    # probability both modalities drop

    # ── Output ──
    [string]$SaveDir = "checkpoints",
    [string]$SaveName = "best.pt",
    [string]$VisDir = "",
    [string]$TbDir = "",
    [string]$Resume = "",

    # ── Device ──
    [string]$CUDA = "",
    [string]$Device = "",
    [int]$NumWorkers = 4,
    [switch]$Compile
)

# ── Cache directories ──
$env:TORCH_HOME = "./cache"
$env:HF_HOME = "./cache"

if (-not [string]::IsNullOrEmpty($CUDA)) {
    $env:CUDA_VISIBLE_DEVICES = $CUDA
}

# ── Build argument list ──
$pyArgs = @(
    "train/run_train.py",
    "--data_root", $DataRoot,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--lr", $LR,
    "--weight_decay", $WeightDecay,
    "--optimizer", $Optimizer,
    "--scheduler", $Scheduler,
    "--warmup_epochs", $WarmupEpochs,
    "--clip_grad", $ClipGrad,
    "--label_smoothing", $LabelSmoothing,
    "--mixup_alpha", $MixupAlpha,
    "--patience", $Patience,
    "--early_stop_metric", $EarlyStopMetric,
    "--seed", $Seed,
    "--save_dir", $SaveDir,
    "--save_name", $SaveName,
    "--num_workers", $NumWorkers
)

# Pipeline
if (-not [string]::IsNullOrEmpty($Pipeline)) {
    $pyArgs += @("--pipeline", $Pipeline)
}

# Custom model / dataset overrides
if (-not [string]::IsNullOrEmpty($ModelModule)) { $pyArgs += @("--model_module", $ModelModule) }
if (-not [string]::IsNullOrEmpty($ModelClass)) { $pyArgs += @("--model_class", $ModelClass) }
if (-not [string]::IsNullOrEmpty($DatasetModule)) { $pyArgs += @("--dataset_module", $DatasetModule) }
if (-not [string]::IsNullOrEmpty($DatasetClass)) { $pyArgs += @("--dataset_class", $DatasetClass) }

# Kwargs
if (-not [string]::IsNullOrEmpty($ModelKwargs)) { $pyArgs += @("--model_kwargs", $ModelKwargs) }
if (-not [string]::IsNullOrEmpty($DatasetKwargs)) { $pyArgs += @("--dataset_kwargs", $DatasetKwargs) }

# IO modes
if (-not [string]::IsNullOrEmpty($InputMode)) { $pyArgs += @("--input_mode", $InputMode) }
if (-not [string]::IsNullOrEmpty($OutputMode)) { $pyArgs += @("--output_mode", $OutputMode) }

# Device
if (-not [string]::IsNullOrEmpty($Device)) { $pyArgs += @("--device", $Device) }

# Resume
if (-not [string]::IsNullOrEmpty($Resume)) { $pyArgs += @("--resume", $Resume) }

# Visualisation
if (-not [string]::IsNullOrEmpty($VisDir)) { $pyArgs += @("--vis_dir", $VisDir) }

# TensorBoard
if (-not [string]::IsNullOrEmpty($TbDir)) { $pyArgs += @("--tb_dir", $TbDir) }

# torch.compile
if ($Compile) { $pyArgs += @("--compile") }

# Corruption / modality dropout
if ($CorruptionMode -ne "none") {
    $pyArgs += @("--corruption_mode", $CorruptionMode)
    $pyArgs += @("--corruption_p_full", $CorruptionPFull)
    $pyArgs += @("--corruption_p_consecutive", $CorruptionPConsecutive)

    $pyArgs += @("--corruption_block_range", $CorruptionBlockRange[0], $CorruptionBlockRange[1])
    $pyArgs += @("--corruption_modalities") + $CorruptionModalities

    $pyArgs += @("--corruption_both_drop_prob", $CorruptionBothDropProb)
}

# ── Run ──
Write-Host "Running: python $($pyArgs -join ' ')" -ForegroundColor Cyan
& python @pyArgs
exit $LASTEXITCODE
