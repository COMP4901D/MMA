"""
datasets/ — Data loading modules for UTD-MHAD.

Three dataset variants:
  UTDMADInertialDataset  — Pure 6-axis IMU
  UTD_MHAD_Dataset       — Depth + IMU (with GAF + segment sampling)
  UTDMADRGBDIMUDataset   — RGB-D + IMU (video + depth + inertial)
"""

# 27 action classes in UTD-MHAD (shared constant)
ACTION_NAMES = [
    "swipe_left", "swipe_right", "wave", "clap", "throw",
    "arm_cross", "basketball_shoot", "draw_x", "draw_circle_CW",
    "draw_circle_CCW", "draw_triangle", "bowling", "boxing",
    "baseball_swing", "tennis_swing", "arm_curl", "tennis_serve",
    "push", "knock", "catch", "pickup_throw",
    "jog", "walk", "sit2stand", "stand2sit", "lunge", "squat",
]

from .transforms import ModalityDropout
from .utd_inertial import UTDMADInertialDataset
from .utd_depth_imu import UTD_MHAD_Dataset
from .utd_rgbd_imu import UTDMADRGBDIMUDataset
from .utd_skel_imu import UTDMADSkelIMUDataset
