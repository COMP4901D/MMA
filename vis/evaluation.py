"""Evaluation & result visualizations.

Confusion matrix, per-class precision/recall/F1, prediction confidence, etc.

Usage:
    from vis.evaluation import plot_confusion_matrix, plot_per_class_metrics
    plot_confusion_matrix(y_true, y_pred, class_names=ACTION_NAMES, save_dir="vis_output")
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------- #
#  Confusion Matrix                                                        #
# ---------------------------------------------------------------------- #

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: List[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_dir: str = None,
    filename: str = "confusion_matrix",
    show: bool = False,
):
    """Plot a confusion matrix heatmap.

    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) predicted labels
        class_names: label names for axes
        normalize: normalize rows to show percentages
    """
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)

    n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_classes), max(7, 0.35 * n_classes)))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, shrink=0.8)

    if class_names:
        tick_labels = [class_names[i] if i < len(class_names) else str(i) for i in range(n_classes)]
    else:
        tick_labels = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate cells (only for small matrices)
    if n_classes <= 30:
        thresh = cm_display.max() / 2.0
        fmt = ".1%" if normalize else "d"
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_display[i, j]
                text = f"{val:{fmt}}" if normalize else f"{int(val)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=5,
                        color="white" if val > thresh else "black")

    fig.tight_layout()
    _save_and_show(fig, save_dir, filename, show)
    return fig


def plot_confusion_matrix_from_array(
    cm,
    class_names: List[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_dir: str = None,
    show: bool = False,
):
    """Plot from a precomputed (n_classes, n_classes) confusion matrix array."""
    cm = _to_numpy(cm)
    n_classes = cm.shape[0]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums
    else:
        cm_display = cm.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_classes), max(7, 0.35 * n_classes)))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    if class_names:
        tick_labels = [class_names[i] if i < len(class_names) else str(i) for i in range(n_classes)]
    else:
        tick_labels = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if n_classes <= 30:
        thresh = cm_display.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_display[i, j]
                text = f"{val:.1%}" if normalize else f"{int(val)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=5,
                        color="white" if val > thresh else "black")

    fig.tight_layout()
    _save_and_show(fig, save_dir, "confusion_matrix", show)
    return fig


# ---------------------------------------------------------------------- #
#  Per-class metrics                                                       #
# ---------------------------------------------------------------------- #

def plot_per_class_metrics(
    y_true,
    y_pred,
    class_names: List[str] = None,
    title: str = "Per-Class Metrics",
    save_dir: str = None,
    show: bool = False,
):
    """Grouped bar chart of precision, recall, F1 per class.

    Args:
        y_true, y_pred: (N,) arrays
        class_names: list of action names
    """
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    n_classes = max(y_true.max(), y_pred.max()) + 1

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision[c] = p
        recall[c] = r
        f1[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    if class_names:
        names = [class_names[i] if i < len(class_names) else str(i) for i in range(n_classes)]
    else:
        names = [str(i) for i in range(n_classes)]

    x = np.arange(n_classes)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * n_classes), 5))
    ax.bar(x - w, precision, w, label="Precision", edgecolor="black", linewidth=0.3)
    ax.bar(x,     recall,    w, label="Recall",    edgecolor="black", linewidth=0.3)
    ax.bar(x + w, f1,        w, label="F1",        edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "per_class_metrics", show)
    return fig


def plot_per_class_accuracy(
    y_true,
    y_pred,
    class_names: List[str] = None,
    title: str = "Per-Class Accuracy",
    save_dir: str = None,
    show: bool = False,
):
    """Horizontal bar chart of per-class accuracy, sorted by accuracy."""
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    n_classes = max(y_true.max(), y_pred.max()) + 1

    acc = np.zeros(n_classes)
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc[c] = (y_pred[mask] == c).sum() / mask.sum()

    order = np.argsort(acc)
    if class_names:
        names = [class_names[i] if i < len(class_names) else str(i) for i in order]
    else:
        names = [str(i) for i in order]

    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * n_classes)))
    colors = plt.cm.RdYlGn(acc[order])
    ax.barh(range(n_classes), acc[order], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.axvline(x=acc.mean(), color="red", linestyle="--", linewidth=0.8, label=f"Mean={acc.mean():.2%}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "per_class_accuracy", show)
    return fig


# ---------------------------------------------------------------------- #
#  Prediction confidence                                                   #
# ---------------------------------------------------------------------- #

def plot_prediction_confidence(
    logits,
    y_true=None,
    title: str = "Prediction Confidence",
    save_dir: str = None,
    show: bool = False,
):
    """Histogram of prediction confidence (softmax max prob).

    Optionally split into correct vs wrong predictions.

    Args:
        logits: (N, C) raw logits or softmax probabilities
        y_true: (N,) ground truth labels
    """
    logits = _to_numpy(logits)
    # Convert to probabilities if not already
    if logits.min() < 0 or logits.max() > 1.01 or not np.allclose(logits.sum(1), 1, atol=0.1):
        # softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    else:
        probs = logits

    max_conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    if y_true is not None:
        y_true = _to_numpy(y_true).astype(int)
        correct = preds == y_true
        ax.hist(max_conf[correct], bins=30, alpha=0.7, label="Correct", edgecolor="black", linewidth=0.3)
        ax.hist(max_conf[~correct], bins=30, alpha=0.7, label="Wrong", edgecolor="black", linewidth=0.3)
        ax.legend()
    else:
        ax.hist(max_conf, bins=30, alpha=0.7, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(x=max_conf.mean(), color="red", linestyle="--", linewidth=0.8,
               label=f"Mean={max_conf.mean():.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "prediction_confidence", show)
    return fig


# ---------------------------------------------------------------------- #
#  Internal helpers                                                        #
# ---------------------------------------------------------------------- #

def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _save_and_show(fig, save_dir, name, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
