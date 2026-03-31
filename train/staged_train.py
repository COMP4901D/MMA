"""
Staged Multi-Modal Trainer
===========================
3-phase training to address modality learning speed asymmetry in RGBD+IMU HAR.

Phase 1 (IMU warmup):   Freeze RGBD encoder, train IMU + fusion + head
Phase 2 (RGBD focus):   Freeze IMU encoder, train RGBD (layer4) + fusion + head
Phase 3 (Joint):        Unfreeze all, differential LR per parameter group

Each phase rebuilds the optimizer and cosine-warmup scheduler.
Logs per-modality auxiliary accuracy for monitoring branch competence.

Trainer interface (called by run_train.py):
    get_criterion(args, **kwargs)
    train_one_epoch(model, loader, optimizer, criterion, device, cfg, args)
    evaluate(model, loader, criterion, device, cfg, args)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ================================================================
#  Phase configuration
# ================================================================

def get_phase_config(cfg):
    """Return phase boundaries and settings from pipeline_cfg."""
    return {
        "phase1_end": cfg.get("phase1_end", 20),
        "phase2_end": cfg.get("phase2_end", 40),
        "phase1_lr": cfg.get("phase1_lr", 3e-4),
        "phase2_lr": cfg.get("phase2_lr", 1e-4),
        "phase3_lr": cfg.get("phase3_lr", 3e-4),
        "phase2_aux_weight": cfg.get("phase2_aux_weight", 0.3),
        "phase3_aux_weight": cfg.get("phase3_aux_weight", 0.3),
        "phase3_rgbd_lr_scale": cfg.get("phase3_rgbd_lr_scale", 0.1),
        "warmup_epochs": cfg.get("staged_warmup_epochs", 5),
    }


def get_current_phase(epoch, phase_cfg):
    """Return 1, 2, or 3 based on epoch (0-indexed)."""
    if epoch < phase_cfg["phase1_end"]:
        return 1
    elif epoch < phase_cfg["phase2_end"]:
        return 2
    return 3


# ================================================================
#  Freeze / unfreeze helpers
# ================================================================

def _freeze_module(module):
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def _unfreeze_module(module):
    """Unfreeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def _apply_phase_freeze(model, phase, phase_cfg):
    """Configure which parameters are frozen for the given phase.
    
    model is the unwrapped MultimodalMMA instance.
    """
    if phase == 1:
        # Freeze RGBD encoder entirely
        _freeze_module(model.rgbd_enc)
        # Freeze RGBD auxiliary head if exists
        if hasattr(model, 'aux_rgbd'):
            _freeze_module(model.aux_rgbd)
        # Train IMU encoder, fusion, head
        _unfreeze_module(model.imu_enc)
        if hasattr(model, 'aux_imu'):
            _unfreeze_module(model.aux_imu)
        # Fusion layers
        _unfreeze_fusion(model)
        # Classification head
        _unfreeze_module(model.head)
        # Modality embeddings (cross_mamba)
        if hasattr(model, 'mod_embed_imu'):
            model.mod_embed_imu.requires_grad = True
        if hasattr(model, 'mod_embed_rgbd'):
            model.mod_embed_rgbd.requires_grad = False

    elif phase == 2:
        # Freeze IMU encoder entirely
        _freeze_module(model.imu_enc)
        if hasattr(model, 'aux_imu'):
            _freeze_module(model.aux_imu)
        # Unfreeze RGBD encoder with partial freeze (layer4 only)
        _unfreeze_module(model.rgbd_enc)
        # Re-apply partial freeze: only layer4+proj trainable
        if hasattr(model.rgbd_enc, 'spatial') and hasattr(model.rgbd_enc.spatial, '_apply_freeze'):
            model.rgbd_enc.spatial._apply_freeze("partial")
        if hasattr(model, 'aux_rgbd'):
            _unfreeze_module(model.aux_rgbd)
        # Override aux_weight for phase 2
        model.aux_weight = phase_cfg["phase2_aux_weight"]
        # Fusion layers
        _unfreeze_fusion(model)
        _unfreeze_module(model.head)
        if hasattr(model, 'mod_embed_rgbd'):
            model.mod_embed_rgbd.requires_grad = True
        if hasattr(model, 'mod_embed_imu'):
            model.mod_embed_imu.requires_grad = False

    elif phase == 3:
        # Unfreeze everything
        _unfreeze_module(model.rgbd_enc)
        _unfreeze_module(model.imu_enc)
        if hasattr(model, 'aux_rgbd'):
            _unfreeze_module(model.aux_rgbd)
        if hasattr(model, 'aux_imu'):
            _unfreeze_module(model.aux_imu)
        _unfreeze_fusion(model)
        _unfreeze_module(model.head)
        # Keep partial freeze on ResNet conv1-layer3
        if hasattr(model.rgbd_enc, 'spatial') and hasattr(model.rgbd_enc.spatial, '_apply_freeze'):
            model.rgbd_enc.spatial._apply_freeze("partial")
        model.aux_weight = phase_cfg["phase3_aux_weight"]
        if hasattr(model, 'mod_embed_rgbd'):
            model.mod_embed_rgbd.requires_grad = True
        if hasattr(model, 'mod_embed_imu'):
            model.mod_embed_imu.requires_grad = True


def _unfreeze_fusion(model):
    """Unfreeze fusion-related modules."""
    if hasattr(model, 'cross_block'):
        _unfreeze_module(model.cross_block)
    if hasattr(model, 'pool'):
        _unfreeze_module(model.pool)
    if hasattr(model, 'rgbd_pool'):
        _unfreeze_module(model.rgbd_pool)
    if hasattr(model, 'imu_pool'):
        _unfreeze_module(model.imu_pool)
    if hasattr(model, 'fusion_layer') and model.fusion_layer is not None:
        _unfreeze_module(model.fusion_layer)


# ================================================================
#  Optimizer builders
# ================================================================

def _build_optimizer(model, phase, phase_cfg, weight_decay):
    """Build AdamW optimizer with phase-appropriate parameter groups and LRs."""
    if phase == 1:
        lr = phase_cfg["phase1_lr"]
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    elif phase == 2:
        lr = phase_cfg["phase2_lr"]
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    else:  # phase 3: differential LR
        base_lr = phase_cfg["phase3_lr"]
        rgbd_lr_scale = phase_cfg["phase3_rgbd_lr_scale"]

        # Collect parameter IDs for RGBD encoder (layer4 + proj)
        rgbd_enc_ids = set()
        for p in model.rgbd_enc.parameters():
            if p.requires_grad:
                rgbd_enc_ids.add(id(p))

        rgbd_params = []
        other_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in rgbd_enc_ids:
                rgbd_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if rgbd_params:
            groups.append({"params": rgbd_params, "lr": base_lr * rgbd_lr_scale})
        if other_params:
            groups.append({"params": other_params, "lr": base_lr})

        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, warmup_epochs, phase_total_epochs):
    """Build a warmup + cosine scheduler for a single phase."""
    class _PhaseCosineWarmup:
        def __init__(self, opt, warmup, total, eta_min=1e-6):
            self.opt = opt
            self.warmup = warmup
            self.total = total
            self.eta_min = eta_min
            self.base_lrs = [pg["lr"] for pg in opt.param_groups]

        def step(self, epoch_in_phase):
            if epoch_in_phase < self.warmup:
                frac = (epoch_in_phase + 1) / self.warmup
                for pg, blr in zip(self.opt.param_groups, self.base_lrs):
                    pg["lr"] = blr * frac
            else:
                progress = (epoch_in_phase - self.warmup) / max(
                    1, self.total - self.warmup)
                for pg, blr in zip(self.opt.param_groups, self.base_lrs):
                    pg["lr"] = self.eta_min + (blr - self.eta_min) * 0.5 * (
                        1 + math.cos(math.pi * progress))

    return _PhaseCosineWarmup(optimizer, warmup_epochs, phase_total_epochs)


# ================================================================
#  Criterion
# ================================================================

def get_criterion(args, **kwargs):
    """Standard cross-entropy with label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


# ================================================================
#  Forward helpers
# ================================================================

def _model_forward(model, batch, device, cfg):
    """Run forward pass, return (output, labels)."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]
    if cfg.get("input_mode") == "list":
        output = model(inputs)
    else:
        output = model(*inputs)
    return output, labels


def _extract_logits(output, cfg):
    """Extract classification logits from model output."""
    mode = cfg.get("output_mode", "logits")
    if mode == "tuple_first":
        return output[0]
    return output


def _mixup_forward(model, batch, device, cfg, criterion, alpha):
    """Mixup forward pass with auxiliary loss support."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]

    lam = np.random.beta(alpha, alpha)
    B = labels.size(0)
    perm = torch.randperm(B, device=device)

    mixed = [lam * x + (1 - lam) * x[perm] for x in inputs]
    if cfg.get("input_mode") == "list":
        output = model(mixed)
    else:
        output = model(*mixed)

    logits = _extract_logits(output, cfg)
    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])

    # Auxiliary loss
    base_model = model.module if hasattr(model, 'module') else model
    if hasattr(base_model, '_aux_logits') and base_model._aux_logits is not None:
        aux_a, aux_b = base_model._aux_logits
        n_aux, aux_sum = 0, 0.0
        if aux_a is not None:
            aux_sum = aux_sum + lam * criterion(aux_a, labels) + (1 - lam) * criterion(aux_a, labels[perm])
            n_aux += 1
        if aux_b is not None:
            aux_sum = aux_sum + lam * criterion(aux_b, labels) + (1 - lam) * criterion(aux_b, labels[perm])
            n_aux += 1
        if n_aux > 0:
            loss = loss + base_model.aux_weight * aux_sum / n_aux
        base_model._aux_logits = None

    return loss, logits, labels, lam, perm


# ================================================================
#  Training epoch
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, cfg, args):
    """Staged training epoch. Manages phase transitions and freeze schedules.
    
    Phase state is tracked via cfg["_staged_state"], initialized on first call.
    The optimizer and scheduler are rebuilt at each phase transition.
    """
    base_model = model.module if hasattr(model, 'module') else model
    epoch = getattr(base_model, 'current_epoch', 0)

    phase_cfg = get_phase_config(cfg)
    phase = get_current_phase(epoch, phase_cfg)

    # Initialize or detect phase transition
    state = cfg.setdefault("_staged_state", {
        "current_phase": 0,
        "optimizer": None,
        "scheduler": None,
    })

    if state["current_phase"] != phase:
        # Phase transition: reconfigure freeze, optimizer, scheduler
        _apply_phase_freeze(base_model, phase, phase_cfg)

        n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        phase_names = {1: "IMU warmup", 2: "RGBD focus", 3: "Joint fine-tune"}
        print(f"\n{'='*60}")
        print(f"  Phase {phase}: {phase_names[phase]}  (epoch {epoch+1})")
        print(f"  Trainable params: {n_trainable:,}")
        print(f"{'='*60}\n")

        # Calculate phase duration for scheduler
        if phase == 1:
            phase_total = phase_cfg["phase1_end"]
        elif phase == 2:
            phase_total = phase_cfg["phase2_end"] - phase_cfg["phase1_end"]
        else:
            phase_total = args.epochs - phase_cfg["phase2_end"]

        opt = _build_optimizer(base_model, phase, phase_cfg, args.weight_decay)
        sched = _build_scheduler(opt, phase_cfg["warmup_epochs"], phase_total)

        state["current_phase"] = phase
        state["optimizer"] = opt
        state["scheduler"] = sched

    # Use the phase-specific optimizer (not the one from run_train.py)
    opt = state["optimizer"]
    sched = state["scheduler"]

    # Compute epoch within current phase for scheduler
    if phase == 1:
        epoch_in_phase = epoch
    elif phase == 2:
        epoch_in_phase = epoch - phase_cfg["phase1_end"]
    else:
        epoch_in_phase = epoch - phase_cfg["phase2_end"]

    sched.step(epoch_in_phase)

    # Training loop
    model.train()
    # Re-apply freeze (train() unfreezes batch norms)
    _apply_phase_freeze(base_model, phase, phase_cfg)

    total_loss, correct, total = 0.0, 0, 0
    aux_rgbd_correct, aux_imu_correct, aux_total = 0, 0, 0

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for batch in loader:
        opt.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            if args.mixup_alpha > 0:
                loss, logits, labels, lam, perm = _mixup_forward(
                    model, batch, device, cfg, criterion, args.mixup_alpha)
            else:
                output, labels = _model_forward(model, batch, device, cfg)
                logits = _extract_logits(output, cfg)
                loss = criterion(logits, labels)
                lam, perm = 1.0, None

        # Auxiliary loss (no mixup path for simplicity)
        if not args.mixup_alpha > 0:
            if hasattr(base_model, '_aux_logits') and base_model._aux_logits is not None:
                aux_a, aux_b = base_model._aux_logits
                n_aux, aux_sum = 0, 0.0
                if aux_a is not None:
                    aux_sum = aux_sum + criterion(aux_a, labels)
                    n_aux += 1
                    aux_rgbd_correct += (aux_a.detach().float().argmax(1) == labels).sum().item()
                if aux_b is not None:
                    aux_sum = aux_sum + criterion(aux_b, labels)
                    n_aux += 1
                    aux_imu_correct += (aux_b.detach().float().argmax(1) == labels).sum().item()
                if n_aux > 0:
                    loss = loss + base_model.aux_weight * aux_sum / n_aux
                aux_total += labels.size(0)
                base_model._aux_logits = None

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        pred = logits.detach().float().argmax(1)
        if perm is not None:
            correct += (lam * (pred == labels).float()
                        + (1 - lam) * (pred == labels[perm]).float()).sum().item()
        else:
            correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total

    # Log per-modality auxiliary accuracies
    if aux_total > 0:
        rgbd_aux_acc = aux_rgbd_correct / aux_total if aux_rgbd_correct > 0 else 0.0
        imu_aux_acc = aux_imu_correct / aux_total if aux_imu_correct > 0 else 0.0
        phase_lr = opt.param_groups[0]["lr"]
        print(f"  [Phase {phase}] AuxRGBD={rgbd_aux_acc:.4f} AuxIMU={imu_aux_acc:.4f} LR={phase_lr:.2e}")

    return avg_loss, avg_acc


# ================================================================
#  Evaluation
# ================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, args):
    """Standard evaluation. Returns dict with loss, acc, prec, rec, f1, preds, labels."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_logits = [], [], []
    use_amp = (device.type == "cuda")

    for batch in loader:
        with autocast(device_type=device.type, enabled=use_amp):
            output, labels = _model_forward(model, batch, device, cfg)
            logits = _extract_logits(output, cfg)
            total_loss += criterion(logits, labels).item() * labels.size(0)
        logits_f = logits.float().cpu().numpy()
        all_logits.append(logits_f)
        all_preds.append(logits_f.argmax(1))
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    all_logits_np = np.concatenate(all_logits)
    n = len(labels)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    return {
        "loss": total_loss / n,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "preds": preds,
        "labels": labels,
        "logits": all_logits_np,
    }
