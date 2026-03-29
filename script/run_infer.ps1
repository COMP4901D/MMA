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

    # ── Infrastructure ──
    [int]$BatchSize   = 32,
    [int]$NumWorkers  = 4,
    [string]$CUDA     = "",
    [string]$Device   = ""
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

# ── Run ──
Write-Host "Running: python $($pyArgs -join ' ')" -ForegroundColor Cyan
& python @pyArgs
exit $LASTEXITCODE
