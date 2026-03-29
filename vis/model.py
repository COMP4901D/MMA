"""Model-level visualizations: weight distributions, gradient flow, feature embeddings.

All functions accept plain numpy arrays or PyTorch models/tensors to stay decoupled.

Usage:
    from vis.model import plot_gradient_flow, plot_param_histogram
    plot_gradient_flow(model.named_parameters(), save_dir="vis_output")
"""

import os
from typing import Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------- #
#  Gradient flow                                                           #
# ---------------------------------------------------------------------- #

def plot_gradient_flow(
    named_parameters: Iterator[Tuple[str, "torch.nn.Parameter"]],
    title: str = "Gradient Flow",
    save_dir: str = None,
    show: bool = False,
):
    """Visualize gradient magnitudes across layers.

    Call after loss.backward() and before optimizer.step().

    Args:
        named_parameters: model.named_parameters() iterator
    """
    names = []
    avg_grads = []
    max_grads = []

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad = param.grad.detach().cpu().numpy()
            names.append(name)
            avg_grads.append(np.abs(grad).mean())
            max_grads.append(np.abs(grad).max())

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(names)), 5))
    x = range(len(names))
    ax.bar(x, max_grads, alpha=0.3, lw=1, color="tab:blue", label="Max |grad|")
    ax.bar(x, avg_grads, alpha=0.7, lw=1, color="tab:orange", label="Mean |grad|")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Gradient|")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "gradient_flow", show)
    return fig


# ---------------------------------------------------------------------- #
#  Parameter distributions                                                 #
# ---------------------------------------------------------------------- #

def plot_param_histogram(
    named_parameters: Iterator[Tuple[str, "torch.nn.Parameter"]],
    max_layers: int = 16,
    title: str = "Weight Distributions",
    save_dir: str = None,
    show: bool = False,
):
    """Histogram of weight values across selected layers.

    Args:
        named_parameters: model.named_parameters()
        max_layers: maximum number of layers to plot (picks evenly spaced)
    """
    all_params = []
    for name, param in named_parameters:
        if param.requires_grad and param.dim() >= 2:
            all_params.append((name, param.detach().cpu().numpy().flatten()))

    if not all_params:
        return None

    if len(all_params) > max_layers:
        indices = np.linspace(0, len(all_params) - 1, max_layers, dtype=int)
        all_params = [all_params[i] for i in indices]

    n = len(all_params)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, (name, values) in enumerate(all_params):
        ax = axes[idx]
        ax.hist(values, bins=50, alpha=0.7, edgecolor="black", linewidth=0.3)
        short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
        ax.set_title(short_name, fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "param_histogram", show)
    return fig


# ---------------------------------------------------------------------- #
#  Feature embeddings (t-SNE)                                              #
# ---------------------------------------------------------------------- #

def plot_feature_tsne(
    features,
    labels,
    class_names: List[str] = None,
    perplexity: float = 30.0,
    title: str = "t-SNE Feature Embedding",
    save_dir: str = None,
    show: bool = False,
):
    """2D t-SNE scatter plot of learned feature representations.

    Args:
        features: (N, D) feature vectors (e.g., from penultimate layer)
        labels: (N,) integer class labels
        class_names: optional list of label names
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[vis] sklearn not available, skipping t-SNE plot.")
        return None

    features = _to_numpy(features)
    labels = _to_numpy(labels).astype(int)

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features) - 1),
                random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(features)

    n_classes = labels.max() + 1
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        lbl = class_names[c] if class_names and c < len(class_names) else str(c)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(c)],
                   label=lbl, s=15, alpha=0.7, edgecolors="none")

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    if n_classes <= 27:
        ax.legend(fontsize=5, ncol=2, loc="best", markerscale=2)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "tsne", show)
    return fig


# ---------------------------------------------------------------------- #
#  Attention weights                                                       #
# ---------------------------------------------------------------------- #

def plot_attention_weights(
    weights,
    x_labels: List[str] = None,
    y_labels: List[str] = None,
    title: str = "Attention Weights",
    save_dir: str = None,
    filename: str = "attention_weights",
    show: bool = False,
):
    """Heatmap of attention weight matrix.

    Args:
        weights: (H, W) or (N_heads, H, W) attention weights
    """
    weights = _to_numpy(weights)
    if weights.ndim == 3:
        n_heads = weights.shape[0]
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = np.atleast_2d(axes).flatten()
        for i in range(n_heads):
            ax = axes[i]
            im = ax.imshow(weights[i], cmap="viridis", aspect="auto")
            ax.set_title(f"Head {i}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.7)
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(weights, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.8)
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, filename, show)
    return fig


def plot_fusion_alpha(
    alpha_history,
    segment_labels: List[str] = None,
    title: str = "Fusion Alpha Over Epochs",
    save_dir: str = None,
    show: bool = False,
):
    """Plot evolution of fusion attention alpha across training epochs.

    Args:
        alpha_history: list of alpha values (one per epoch), each can be
                      scalar or (N_seg,) array.
    """
    alpha_history = [_to_numpy(a) for a in alpha_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    if np.ndim(alpha_history[0]) == 0:
        ax.plot(alpha_history, marker=".", markersize=3)
        ax.set_ylabel("Alpha")
    else:
        arr = np.array(alpha_history)
        for i in range(arr.shape[1]):
            lbl = segment_labels[i] if segment_labels and i < len(segment_labels) else f"Seg {i}"
            ax.plot(arr[:, i], label=lbl, marker=".", markersize=2)
        ax.legend(fontsize=8)
        ax.set_ylabel("Alpha")

    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "fusion_alpha", show)
    return fig


# ---------------------------------------------------------------------- #
#  Model summary                                                           #
# ---------------------------------------------------------------------- #

def plot_model_size(
    named_parameters: Iterator[Tuple[str, "torch.nn.Parameter"]],
    top_k: int = 20,
    title: str = "Layer Parameter Counts",
    save_dir: str = None,
    show: bool = False,
):
    """Horizontal bar chart of parameter counts per layer.

    Args:
        named_parameters: model.named_parameters()
        top_k: show only the top-k largest layers
    """
    layer_sizes = []
    for name, param in named_parameters:
        layer_sizes.append((name, param.numel()))

    layer_sizes.sort(key=lambda x: x[1], reverse=True)
    layer_sizes = layer_sizes[:top_k]
    layer_sizes.reverse()

    names = [n for n, _ in layer_sizes]
    sizes = [s for _, s in layer_sizes]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(names))))
    ax.barh(range(len(names)), sizes, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Parameter Count")
    ax.set_title(title)
    total = sum(s for _, s in layer_sizes)
    ax.text(0.98, 0.02, f"Shown: {total:,} params",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic")
    fig.tight_layout()
    _save_and_show(fig, save_dir, "model_size", show)
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
