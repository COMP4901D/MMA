"""Training metrics logging and visualization.

Usage:
    logger = TrainingLogger()
    for epoch in range(n_epochs):
        logger.log(epoch, train_loss=0.5, train_acc=0.8, val_loss=0.4, val_acc=0.85)
    logger.plot_curves(save_dir="vis_output")
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """Collects per-epoch metrics and generates plots.

    Designed to be pipeline-agnostic: just call logger.log(epoch, **metrics).
    """

    def __init__(self):
        self.history = defaultdict(list)
        self.epochs = []

    def log(self, epoch: int, **metrics):
        """Record metrics for one epoch.

        Args:
            epoch: epoch number (0-based or 1-based, your choice)
            **metrics: arbitrary key-value pairs, e.g.
                train_loss=0.5, val_acc=0.85, lr=1e-4
        """
        self.epochs.append(epoch)
        for k, v in metrics.items():
            self.history[k].append(float(v))

    def save_json(self, path: str):
        """Persist full training history to JSON."""
        data = {"epochs": self.epochs, **{k: v for k, v in self.history.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(path: str) -> "TrainingLogger":
        """Reconstruct a logger from a saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger = TrainingLogger()
        logger.epochs = data.pop("epochs")
        for k, v in data.items():
            logger.history[k] = v
        return logger

    # ------------------------------------------------------------------ #
    #  Plotting helpers                                                    #
    # ------------------------------------------------------------------ #

    def plot_curves(self, save_dir: str = None, show: bool = False):
        """Auto-detect metric groups and create subplots.

        Groups by prefix: train_* / val_* / test_* are paired.
        Standalone keys get their own subplot.
        """
        groups = _group_metrics(list(self.history.keys()))
        n = len(groups)
        if n == 0:
            return None

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes.flatten()

        for ax, (title, keys) in zip(axes, groups.items()):
            for key in keys:
                label = key.replace("_", " ").title()
                ax.plot(self.epochs, self.history[key], label=label)
            ax.set_xlabel("Epoch")
            ax.set_title(title.replace("_", " ").title())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        _save_and_show(fig, save_dir, "training_curves", show)
        return fig

    def plot_loss(self, save_dir: str = None, show: bool = False):
        """Plot only loss-related metrics."""
        return self._plot_subset("loss", save_dir, show)

    def plot_accuracy(self, save_dir: str = None, show: bool = False):
        """Plot only accuracy-related metrics."""
        return self._plot_subset("acc", save_dir, show)

    def plot_lr(self, save_dir: str = None, show: bool = False):
        """Plot learning rate schedule."""
        return self._plot_subset("lr", save_dir, show)

    def plot_momentum_params(self, save_dir: str = None, show: bool = False):
        """Plot momentum alpha/beta evolution."""
        keys = [k for k in self.history if k.lower() in ("alpha", "beta", "α", "β")]
        if not keys:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in keys:
            ax.plot(self.epochs, self.history[k], label=k)
        ax.set_xlabel("Epoch")
        ax.set_title("Momentum Parameters")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_and_show(fig, save_dir, "momentum_params", show)
        return fig

    def _plot_subset(self, keyword, save_dir, show):
        keys = [k for k in self.history if keyword in k.lower()]
        if not keys:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in keys:
            ax.plot(self.epochs, self.history[k], label=k.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_title(keyword.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_and_show(fig, save_dir, keyword, show)
        return fig


# ---------------------------------------------------------------------- #
#  Standalone functions (work without TrainingLogger)                      #
# ---------------------------------------------------------------------- #


def plot_training_curves(
    epochs,
    train_loss=None, val_loss=None,
    train_acc=None, val_acc=None,
    title="Training Curves",
    save_dir=None, show=False,
):
    """Quick plot of loss and/or accuracy curves.

    Args:
        epochs: list/array of epoch numbers
        train_loss, val_loss: optional loss arrays
        train_acc, val_acc: optional accuracy arrays
        title: figure title
        save_dir: directory to save PNG (None = don't save)
        show: whether to call plt.show()

    Returns:
        matplotlib Figure
    """
    has_loss = train_loss is not None or val_loss is not None
    has_acc = train_acc is not None or val_acc is not None
    ncols = int(has_loss) + int(has_acc)
    if ncols == 0:
        return None

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_loss:
        ax = axes[idx]; idx += 1
        if train_loss is not None:
            ax.plot(epochs, train_loss, label="Train Loss")
        if val_loss is not None:
            ax.plot(epochs, val_loss, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    if has_acc:
        ax = axes[idx]
        if train_acc is not None:
            ax.plot(epochs, train_acc, label="Train Acc")
        if val_acc is not None:
            ax.plot(epochs, val_acc, label="Val Acc")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "training_curves", show)
    return fig


def plot_lr_schedule(epochs, lr_values, save_dir=None, show=False):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(epochs, lr_values, color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "lr_schedule", show)
    return fig


# ---------------------------------------------------------------------- #
#  Internal helpers                                                        #
# ---------------------------------------------------------------------- #

_PREFIXES = ("train_", "val_", "test_", "tr_", "te_")


def _group_metrics(keys):
    """Group metrics by suffix: train_loss & val_loss → 'loss' group."""
    groups = defaultdict(list)
    standalone = []
    for k in keys:
        matched = False
        for prefix in _PREFIXES:
            if k.lower().startswith(prefix):
                suffix = k[len(prefix):]
                groups[suffix].append(k)
                matched = True
                break
        if not matched:
            standalone.append(k)

    # pair up: only keep groups with >1 member; demote singletons
    result = {}
    for suffix, ks in groups.items():
        result[suffix] = ks
    for k in standalone:
        result[k] = [k]
    return result


def _save_and_show(fig, save_dir, name, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
