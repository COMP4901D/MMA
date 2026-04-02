# CMAR Round-2 Sweep: Focused on RGBD-only > 0.50
# Baseline: CMAR5 (imu=0.50, rgbd=0.05, cmar=0.1) → RGBD-only=0.4860
# Strategy: vary imu dropout (0.50-0.60), rgbd dropout (0.00-0.05), cmar (0.1-0.2)

$common = @(
    "--pipeline", "rgbd_imu",
    "--data_root", "./datasets/UTD-MHAD",
    "--epochs", "100",
    "--tb_dir", "runs",
    "--num_workers", "0",
    "--batch_size", "32",
    "--label_smoothing", "0.1",
    "--mixup_alpha", "0.15",
    "--weight_decay", "0.008",
    "--scheduler", "cosine_warmup",
    "--warmup_epochs", "5",
    "--lr", "3e-4",
    "--patience", "40",
    "--seed", "42",
    "--dataset_kwargs", '{"augment":true}'
)

$experiments = @(
    # CMAR12: Push imu dropout to 0.55 (CMAR5 had 0.50)
    @{ name="CMAR12"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.55,"md_drop_rgbd":0.05,"cmar_weight":0.1,"cmar_proj_dim":64}' },

    # CMAR13: Aggressive imu dropout 0.60
    @{ name="CMAR13"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.60,"md_drop_rgbd":0.05,"cmar_weight":0.1,"cmar_proj_dim":64}' },

    # CMAR14: Remove rgbd dropout entirely (test if rgbd=0.05 helped or hurt)
    @{ name="CMAR14"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.50,"md_drop_rgbd":0.00,"cmar_weight":0.1,"cmar_proj_dim":64}' },

    # CMAR15: Higher CMAR weight at sweet spot (test if stronger alignment helps)
    @{ name="CMAR15"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.50,"md_drop_rgbd":0.05,"cmar_weight":0.2,"cmar_proj_dim":64}' },

    # CMAR16: Combined: higher imu + slightly higher cmar
    @{ name="CMAR16"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.55,"md_drop_rgbd":0.03,"cmar_weight":0.15,"cmar_proj_dim":64}' }
)

Write-Host "============================================"
Write-Host "  CMAR Round-2 Sweep: 5 Experiments"
Write-Host "  Target: RGBD-only > 0.50"
Write-Host "============================================"

foreach ($exp in $experiments) {
    $name = $exp.name
    $kwargs = $exp.kwargs
    $saveName = "rgbd_imu_${name}.pt"

    Write-Host ""
    Write-Host ">>> Starting $name  (save: $saveName)"
    Write-Host "    model_kwargs: $kwargs"
    Write-Host ""

    python train/run_train.py @common `
        --save_name $saveName `
        --model_kwargs $kwargs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!! $name FAILED (exit code $LASTEXITCODE) — skipping"
        continue
    }

    Write-Host ""
    Write-Host ">>> Evaluating $name (eval_missing all)"
    Write-Host ""

    python infer/run_infer.py `
        --pipeline rgbd_imu `
        --data_root "./datasets/UTD-MHAD" `
        --checkpoint "checkpoints/$saveName" `
        --num_workers 0 `
        --eval_missing all

    Write-Host ""
    Write-Host "=== $name COMPLETE ==="
    Write-Host ""
}

Write-Host ""
Write-Host "============================================"
Write-Host "  All Round-2 experiments complete!"
Write-Host "============================================"
