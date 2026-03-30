"""
MuMu Trainer
=============
Cooperative multitask training for the MuMu model.

Special features vs. default trainer:
  - Cooperative loss: L = L_target + β · L_aux  (Section 3.4, Eq. 10)
  - Automatically derives activity-group labels from fine-grained labels
  - Extracts target logits (output[1]) for accuracy tracking
  - No mixup support (cooperative loss structure is incompatible with simple mixup)

Reference: Islam & Iqbal, "MuMu: Cooperative Multitask Learning-Based
           Guided Multimodal Fusion", AAAI 2022.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ── UTD-MHAD 5-group mapping ───────────────────────────────────
# Group 0: Arm gestures     (a1-a6)
# Group 1: Sports/full-body (a7, a12-a15, a17)
# Group 2: Hand/wrist draw  (a8-a11)
# Group 3: Interact/push    (a16, a18-a21)
# Group 4: Lower-body/legs  (a22-a27)
UTD_ACTIVITY_TO_GROUP = [
    0, 0, 0, 0, 0, 0,        # a1-a6
    1,                        # a7
    2, 2, 2, 2,               # a8-a11
    1, 1, 1, 1,               # a12-a15
    3, 1, 3, 3, 3, 3,         # a16-a21 (a17→1)
    4, 4, 4, 4, 4, 4,         # a22-a27
]


def activity_to_group(labels: torch.Tensor,
                      mapping: list = None) -> torch.Tensor:
    """Convert fine-grained activity labels → coarse group labels."""
    if mapping is None:
        mapping = UTD_ACTIVITY_TO_GROUP
    lut = torch.tensor(mapping, dtype=torch.long, device=labels.device)
    return lut[labels]


# ── Cooperative Loss ────────────────────────────────────────────

class MuMuCooperativeLoss(nn.Module):
    """L = L_target + β · L_aux.

    Automatically derives group labels from fine-grained labels.
    """

    def __init__(self, beta_aux: float = 0.5,
                 label_smoothing: float = 0.0,
                 group_mapping: list = None):
        super().__init__()
        self.beta_aux = beta_aux
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.group_mapping = group_mapping or UTD_ACTIVITY_TO_GROUP

    def forward(self, output, target_labels):
        """
        Args:
            output: tuple (y_aux, y_target, alpha, attn_weights) from MuMu.
            target_labels: [B] fine-grained activity labels (0-indexed).
        Returns: scalar loss.
        """
        y_aux, y_target = output[0], output[1]
        group_labels = activity_to_group(target_labels, self.group_mapping)
        loss_target = self.ce(y_target, target_labels)
        loss_aux = self.ce(y_aux, group_labels)
        return loss_target + self.beta_aux * loss_aux


# ── Interface functions ─────────────────────────────────────────

def get_criterion(args, **kwargs):
    """Build MuMu cooperative loss."""
    beta = kwargs.get("beta_aux", 0.5)
    group_mapping = kwargs.get("group_mapping", None)
    return MuMuCooperativeLoss(
        beta_aux=beta,
        label_smoothing=args.label_smoothing,
        group_mapping=group_mapping,
    )


def _model_forward(model, batch, device, cfg):
    """MuMu forward: model(inputs_as_list) → (y_aux, y_target, alpha, attn)."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]
    if cfg.get("input_mode") == "list":
        output = model(inputs)
    else:
        output = model(*inputs)
    return output, labels


def train_one_epoch(model, loader, optimizer, criterion, device, cfg, args):
    """MuMu training epoch with cooperative loss."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for batch in loader:
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            output, labels = _model_forward(model, batch, device, cfg)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        # Track target-task accuracy
        y_target = output[1]
        total_loss += loss.item() * labels.size(0)
        correct += (y_target.detach().float().argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, args):
    """MuMu evaluation with cooperative loss."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    use_amp = (device.type == "cuda")

    for batch in loader:
        with autocast(device_type=device.type, enabled=use_amp):
            output, labels = _model_forward(model, batch, device, cfg)
            loss = criterion(output, labels)

        y_target = output[1]
        total_loss += loss.item() * labels.size(0)
        all_preds.append(y_target.float().argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
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
    }
