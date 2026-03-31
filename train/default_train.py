"""
Default Trainer
===============
Generic training / evaluation functions used by most pipelines
(utdmad, mmtsa, rgbd_imu, etc.).

Trainer modules expose a standard interface that run_train.py calls:
    get_criterion(args, **kwargs)  -> nn.Module
    train_one_epoch(model, loader, optimizer, criterion, device, cfg, args) -> (loss, acc)
    evaluate(model, loader, criterion, device, cfg, args) -> metrics_dict
    mixup_forward(model, batch, device, cfg, criterion, alpha) -> (loss, logits, labels)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ================================================================
#  Forward helpers
# ================================================================

def model_forward(model, batch, device, cfg):
    """Run forward pass. Returns (output, labels)."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]
    if cfg.get("input_mode") == "list":
        output = model(inputs)
    else:
        output = model(*inputs)
    return output, labels


def extract_logits(output, cfg):
    """Extract classification logits from model output."""
    mode = cfg.get("output_mode", "logits")
    if mode == "tuple_first":
        return output[0]
    return output  # "logits"


# ================================================================
#  Criterion
# ================================================================

def get_criterion(args, **kwargs):
    """Standard cross-entropy with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


# ================================================================
#  Mixup
# ================================================================

def mixup_forward(model, batch, device, cfg, criterion, alpha):
    """Apply mixup to all inputs, return (loss, logits, labels)."""
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

    logits = extract_logits(output, cfg)
    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])

    # Auxiliary multi-task loss in mixup path
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

    # CMAR loss (cross-modal alignment regularization)
    if hasattr(base_model, '_cmar_loss') and base_model._cmar_loss is not None:
        loss = loss + base_model.cmar_weight * base_model._cmar_loss
        base_model._cmar_loss = None

    return loss, logits, labels, lam, perm


# ================================================================
#  Training
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, cfg, args):
    """Generic training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for batch in loader:
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            if args.mixup_alpha > 0:
                loss, logits, labels, lam, perm = mixup_forward(
                    model, batch, device, cfg, criterion, args.mixup_alpha,
                )
            else:
                output, labels = model_forward(model, batch, device, cfg)

                # Simultaneous modality dropout: output is a dict
                if isinstance(output, dict):
                    logits = output["logits"]
                    loss = criterion(logits, labels)
                    md_lam = output["md_lambda"]
                    loss = loss + md_lam * criterion(output["logits_rgbd_only"], labels)
                    loss = loss + md_lam * criterion(output["logits_imu_only"], labels)
                else:
                    logits = extract_logits(output, cfg)
                    loss = criterion(logits, labels)
                lam, perm = 1.0, None

        # Auxiliary multi-task loss (per-modality heads)
        base_model = model.module if hasattr(model, 'module') else model
        if hasattr(base_model, '_aux_logits') and base_model._aux_logits is not None:
            aux_a, aux_b = base_model._aux_logits
            n_aux, aux_sum = 0, 0.0
            if aux_a is not None:
                aux_sum = aux_sum + criterion(aux_a, labels)
                n_aux += 1
            if aux_b is not None:
                aux_sum = aux_sum + criterion(aux_b, labels)
                n_aux += 1
            if n_aux > 0:
                loss = loss + base_model.aux_weight * aux_sum / n_aux
            base_model._aux_logits = None

        # CMAR loss (cross-modal alignment regularization)
        if hasattr(base_model, '_cmar_loss') and base_model._cmar_loss is not None:
            loss = loss + base_model.cmar_weight * base_model._cmar_loss
            base_model._cmar_loss = None

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        pred = logits.detach().float().argmax(1)
        if perm is not None:
            # Weighted accuracy: credit prediction if it matches either mixed label
            correct += (lam * (pred == labels).float()
                        + (1 - lam) * (pred == labels[perm]).float()).sum().item()
        else:
            correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ================================================================
#  Evaluation
# ================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, args):
    """Generic evaluation. Returns dict with loss, acc, prec, rec, f1, preds, labels, logits."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_logits = [], [], []
    use_amp = (device.type == "cuda")

    for batch in loader:
        with autocast(device_type=device.type, enabled=use_amp):
            output, labels = model_forward(model, batch, device, cfg)
            logits = extract_logits(output, cfg)
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
        labels, preds, average="weighted", zero_division=0,
    )
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
