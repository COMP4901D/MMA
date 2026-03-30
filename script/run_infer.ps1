<# .SYNOPSIS
  Unified inference launcher for all MMA HAR pipelines.

.DESCRIPTION
  Calls infer/run_infer.py with the chosen pipeline, checkpoint, and parameters.
  If the checkpoint was saved by the unified trainer, pipeline/model_kwargs
  are restored automatically (override with explicit args).

.EXAMPLE
  # Evaluate MMTSA checkpoint
  .\script\run_infer.ps1 -Pipeline mmtsa -Checkpoint checkpoints/best.pt

  # Evaluate with visualisation output
  .\script\run_infer.ps1 -Pipeline rgbd_imu -Checkpoint best.pt -VisDir vis_results

  # Custom model inference
  .\script\run_infer.ps1 -ModelModule "model.mma_utdmad" -ModelClass "MomentumMambaHAR" `
      -DatasetModule "datasets.utd_inertial" -DatasetClass "UTDMADInertialDataset" `
      -DataRoot "./datasets/UTD-MHAD/Inertial" -Checkpoint best.pt `
      -InputMode unpack -OutputMode logits

  # Robustness evaluation: full modality dropout on test set
  .\script\run_infer.ps1 -Pipeline rgbd_imu -Checkpoint checkpoints/best.pt `
      -CorruptionMode full -CorruptionPFull 0.3

  # Robustness evaluation: consecutive dropout on IMU only
  .\script\run_infer.ps1 -Pipeline rgbd_imu -Checkpoint checkpoints/best.pt `
      -CorruptionMode consecutive -CorruptionPConsecutive 0.5 `
      -CorruptionModalities "imu" -VisDir vis_corrupt
#>

param(
    # ── Pipeline selection ──
    [string]$Pipeline = "",               # utdmad | mmtsa | rgbd_imu | mumu | "" for custom/auto

    # ── Custom model / dataset ──
    [string]$ModelModule  = "",
    [string]$ModelClass   = "",
    [string]$DatasetModule = "",
    [string]$DatasetClass  = "",

    # ── Kwargs (JSON) ──
    [string]$ModelKwargs   = "",
    [string]$DatasetKwargs = "",

    # ── Data & checkpoint ──
    [string]$DataRoot    = "./datasets/UTD-MHAD",
    [string]$Checkpoint  = "checkpoints/best.pt",

    # ── IO modes ──
    [string]$InputMode  = "",
    [string]$OutputMode = "",

    # ── Output ──
    [string]$Output  = "preds.npz",
    [string]$VisDir  = "",

    # ── Corruption / modality dropout (robustness evaluation) ──
    [string]$CorruptionMode = "none",        # none | full | consecutive | mixed
    [double]$CorruptionPFull = 0.2,           # per-modality full-dropout probability
    [double]$CorruptionPConsecutive = 0.3,    # per-timestep block-start probability
    [int[]]$CorruptionBlockRange = @(2, 6),    # (min, max) consecutive block length
    [string[]]$CorruptionModalities = @("rgbd", "imu"),  # eligible modalities
    [double]$CorruptionBothDropProb = 0.0,    # probability both modalities drop

    # ── Infrastructure ──
    [int]$BatchSize   = 32,
    [int]$NumWorkers  = 4,
    [string]$CUDA     = "",
    [string]$Device   = "",
    [switch]$Compile
)

# ── Cache ──
$env:TORCH_HOME = "./cache"
$env:HF_HOME    = "./cache"

if (-not [string]::IsNullOrEmpty($CUDA)) {
    $env:CUDA_VISIBLE_DEVICES = $CUDA
}

# ── Build argument list ──
$pyArgs = @(
    "infer/run_infer.py",
    "--data_root", $DataRoot,
    "--checkpoint", $Checkpoint,
    "--batch_size", $BatchSize,
    "--num_workers", $NumWorkers,
    "--output", $Output
)

if (-not [string]::IsNullOrEmpty($Pipeline))      { $pyArgs += @("--pipeline", $Pipeline) }
if (-not [string]::IsNullOrEmpty($ModelModule))    { $pyArgs += @("--model_module", $ModelModule) }
if (-not [string]::IsNullOrEmpty($ModelClass))     { $pyArgs += @("--model_class", $ModelClass) }
if (-not [string]::IsNullOrEmpty($DatasetModule))  { $pyArgs += @("--dataset_module", $DatasetModule) }
if (-not [string]::IsNullOrEmpty($DatasetClass))   { $pyArgs += @("--dataset_class", $DatasetClass) }
if (-not [string]::IsNullOrEmpty($ModelKwargs))    { $pyArgs += @("--model_kwargs", $ModelKwargs) }
if (-not [string]::IsNullOrEmpty($DatasetKwargs))  { $pyArgs += @("--dataset_kwargs", $DatasetKwargs) }
if (-not [string]::IsNullOrEmpty($InputMode))      { $pyArgs += @("--input_mode", $InputMode) }
if (-not [string]::IsNullOrEmpty($OutputMode))     { $pyArgs += @("--output_mode", $OutputMode) }
if (-not [string]::IsNullOrEmpty($Device))         { $pyArgs += @("--device", $Device) }
if (-not [string]::IsNullOrEmpty($VisDir))         { $pyArgs += @("--vis_dir", $VisDir) }

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
