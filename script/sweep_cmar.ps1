# CMAR Hyperparameter Sweep (CMAR2–CMAR11)
# Base: d_model=160, cross_mamba, encoder=pretrained, freeze=partial, temporal_velocity=true
# All other training params identical to CMAR1

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
    @{ name="CMAR2";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.45,"md_drop_rgbd":0.10,"cmar_weight":0.1,"cmar_proj_dim":64}' },
    @{ name="CMAR3";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.35,"md_drop_rgbd":0.10,"cmar_weight":0.3,"cmar_proj_dim":64}' },
    @{ name="CMAR4";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.45,"md_drop_rgbd":0.10,"cmar_weight":0.3,"cmar_proj_dim":64}' },
    @{ name="CMAR5";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.50,"md_drop_rgbd":0.05,"cmar_weight":0.1,"cmar_proj_dim":64}' },
    @{ name="CMAR6";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.45,"md_drop_rgbd":0.10,"cmar_weight":0.1,"cmar_proj_dim":128}' },
    @{ name="CMAR7";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.3,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.35,"md_drop_rgbd":0.10,"cmar_weight":0.1,"cmar_proj_dim":64}' },
    @{ name="CMAR8";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.45,"md_drop_rgbd":0.05,"cmar_weight":0.3,"cmar_proj_dim":64}' },
    @{ name="CMAR9";  kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.50,"md_drop_rgbd":0.10,"cmar_weight":0.2,"cmar_proj_dim":64}' },
    @{ name="CMAR10"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.2,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.45,"md_drop_rgbd":0.10,"cmar_weight":0.2,"cmar_proj_dim":64}' },
    @{ name="CMAR11"; kwargs='{"d_model":160,"fusion":"cross_mamba","aux_weight":0.1,"encoder":"pretrained","freeze":"partial","temporal_velocity":true,"md_schedule":"none","md_drop_imu":0.40,"md_drop_rgbd":0.05,"cmar_weight":0.5,"cmar_proj_dim":128}' }
)

Write-Host "============================================"
Write-Host "  CMAR Hyperparameter Sweep: 10 Experiments"
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

    Write-Host ""
    Write-Host ">>> $name training complete. Running eval..."
    Write-Host ""

    python infer/run_infer.py `
        --pipeline rgbd_imu `
        --data_root "datasets/UTD-MHAD" `
        --checkpoint "checkpoints/$saveName" `
        --num_workers 0 `
        --eval_missing all

    Write-Host ""
    Write-Host "============================================"
}

Write-Host ""
Write-Host "All 10 experiments complete!"
