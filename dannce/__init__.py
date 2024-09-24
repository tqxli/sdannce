from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ConfigBase:
    # Config file paths
    base_config: str | None = None
    io_config: str | None = None
    slurm_config: str | None = None

    # Experiment
    sbatch: bool = False
    random_seed: int | None = None
    gpu_id: str = "0"
    multi_gpu_train: bool = False

    # Videos
    viddir: str = "videos"
    viddir_flag: bool = False
    immode: Literal["vid", "img"] = "vid"
    chunks: Any | None = None
    lockfirst: bool = False
    camnames: list | None = None
    extension: str | None = ".mp4"
    n_views: int | None = 6
    mono: bool = False
    mirror: bool = False

    # Images
    n_channels_in: int | None = 3
    raw_im_h: int | None = None
    raw_im_w: int | None = None
    crop_height: int | None = None
    crop_width: int | None = None
    downfac: int | None = None

    # Dataset
    n_instances: int = 1
    use_npy: bool = True

    # Model
    net: str = ""
    net_type: str = "compressed_dannce"
    batch_size: int = 4

    # Debug
    write_npy: bool = False
    write_visual_hull: bool | None = None
    debug_volume_tifdir: str | None = None
    debug_train_volume_tifdir: str | None = None


@dataclass
class ConfigBaseTrain:
    # Experiments included for training
    exp: Any | None = None

    # Dataset, data split
    data_split_seed: int = 42
    training_fraction: float | None = None
    load_valid: str | None = None
    valid_exp: list | None = None
    num_validation_per_exp: int | None = 4
    num_train_per_exp: int | None = None

    drop_landmark: list | None = None

    augment_hue: bool = False
    augment_brightness: bool = False
    augment_hue_val: float = 0.05
    augment_bright_val: float = 0.05
    augment_rotation_val: int = 5

    # Training
    train_mode: Literal["new", "finetune", "continued"] = "new"
    epochs: int = 50
    save_period: int = 10
    lr: float = 1e-4
    lr_scheduler: str | None = None
    loss: str = "L1Loss"
    metric: str | None = None
    norm_method: Literal["layer", "instance", "batch"] = "layer"
    graph_cfg: dict | None = None


@dataclass
class ConfigBasePredict:
    # Prediction range
    max_num_samples: int | str = "max"
    start_sample: int = 0
    start_batch: int = 0

    predict_labeled_only: bool = False
    label3d_index: int = 0


@dataclass
class DANNCEDataset:
    # Basic information
    dataset: str = "label3d"
    dataset_args: Any | None = None
    skeleton: str = "rat23"
    is_social_dataset: bool = False

    # Preprocessing for center of mass (COM)
    comthresh: int = 0
    weighted: bool = False
    com_method: Literal["median", "mean"] = "median"
    com_fromlabels: bool = False
    cthresh: int | None = None
    medfilt_window: int | None = None

    # Preprocessing for 3D volumes
    depth: bool = False
    channel_combo: list | None = None
    cam3_train: bool = False

    use_silhouette: bool = False
    use_silhouette_in_volume: bool = False
    downscale_occluded_view: bool = False

    # Construction of 3D volumes from multi-view images
    vmin: int | float | None = -120
    vmax: int | float | None = 120
    nvox: int | None = 80
    vol_size: int | float | None = None
    expval: bool = True
    sigma: int = 10
    interp: Literal["nearest", "bilinear"] = "nearest"

    # Dataset augmentation
    use_temporal: bool = False
    support_exp: list | None = None
    n_support_chunks: int | None = None
    unlabeled_temp: int = 0


@dataclass
class ConfigDANNCEBase(ConfigBase, DANNCEDataset):
    n_channels_out: int = 20
    new_n_channels_out: int | None = None


@dataclass
class ConfigDANNCETrain(ConfigDANNCEBase, ConfigBaseTrain):
    # Basic information
    dannce_train_dir: str = "DANNCE/train"
    dannce_finetune_weights: str | None = None

    # Volume augmentation
    augment_continuous_rotation: bool = False
    mirror_augmentation: bool = False
    right_keypoints: list | None = None
    left_keypoints: list | None = None
    rand_view_replace: bool = False
    allow_valid_replace: bool = False
    n_rand_views: int = 0
    rotate: bool = True
    heatmap_reg: bool = False
    heatmap_reg_coeff: float = 0.01

    replace_view: int | None = None

    COM_augmentation: dict | None = None
    unlabeled_sampling: Any = "equal"
    form_batch: bool = False
    form_bs: int | None = None

    # TO BE DEPRECATED
    unlabeled_fraction: float | None = None


@dataclass
class ConfigDANNCEPredict(ConfigDANNCEBase, ConfigBasePredict):
    com_file: str | None = None
    dannce_predict_dir: str = "DANNCE/predict"
    dannce_predict_model: str | None = None


@dataclass
class ConfigCOMBase(ConfigBase):
    # Experiments included for training
    com_exp: Any | None = None

    # Target creation
    dsmode: Literal["dsm", "nn"] = "nn"
    sigma: int = 30

    debug: bool = False
    com_debug: bool | None = None


@dataclass
class COMAugmentation:
    augment_rotation: bool = False
    augment_shear: bool = False
    augment_zoom: bool = False
    augment_shear_val: int = 5
    augment_zoom_val: float = 0.05
    augment_shift_val: float = 0.05


@dataclass
class ConfigCOMTrain(ConfigBaseTrain, ConfigCOMBase, COMAugmentation):
    # Basic information
    com_train_dir: str = "COM/train"
    com_finetune_weights: str | None = None

    # Training
    lr: float = 5e-5
    lr_scheduler: str | None = None
    net: str = "unet2d_fullbn"
    n_channels_out: int = 1


@dataclass
class ConfigCOMPredict(ConfigBasePredict, ConfigCOMBase):
    com_predict_dir: str = "COM/predict"
    com_predict_weights: str | None = None


_param_defaults_shared = ConfigBase().__dict__

_param_defaults_dannce = {
    **ConfigDANNCEPredict().__dict__,
    **ConfigDANNCETrain().__dict__,
}

_param_defaults_com = {
    **ConfigCOMPredict().__dict__,
    **ConfigCOMTrain().__dict__,
}
