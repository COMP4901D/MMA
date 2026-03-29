"""Visualization toolkit for MMA HAR project.

Decoupled, reusable plotting functions organized into four modules:

- vis.training       : Training curves (loss, accuracy, LR, momentum params)
- vis.data_inspection: Data validation (IMU signals, GAF images, depth/RGBD frames)
- vis.evaluation     : Result analysis (confusion matrix, per-class metrics, confidence)
- vis.model          : Model analysis (gradient flow, weight distributions, t-SNE, attention)
"""

from .training import TrainingLogger, plot_training_curves, plot_lr_schedule
from .evaluation import (
    plot_confusion_matrix,
    plot_confusion_matrix_from_array,
    plot_per_class_metrics,
    plot_per_class_accuracy,
    plot_prediction_confidence,
)
from .data_inspection import (
    plot_imu_signal,
    plot_imu_comparison,
    plot_gaf_image,
    plot_gaf_segments,
    plot_depth_frames,
    plot_depth_representation,
    plot_rgbd_frames,
    plot_batch_overview,
    plot_data_distribution,
    plot_class_distribution,
)
from .model import (
    plot_gradient_flow,
    plot_param_histogram,
    plot_feature_tsne,
    plot_attention_weights,
    plot_fusion_alpha,
    plot_model_size,
)
