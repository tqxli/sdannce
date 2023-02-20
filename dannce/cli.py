"""Entrypoints for dannce training and prediction."""
from dannce.interface import (
    com_predict,
    com_train,
    dannce_predict,
    dannce_train,
    sdannce_predict,
    sdannce_train,
)
from dannce.config import check_config, infer_params, build_params
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)
import os
import sys
import ast
import argparse
import yaml
from typing import Dict, Text
import subprocess
from cluster.multi_gpu import MultiGpuHandler


def load_params(param_path: Text) -> Dict:
    """Load a params file

    Args:
        param_path (Text): Path to .yaml file

    Returns:
        Dict: Parameters
    """
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params


def submit_job(args):
    job_name = f"{args.mode}_{args.command}"
    base_config = load_params(args.base_config)
    if "slurm_config" not in base_config:
        raise ValueError("slurm_config not specified in base_config.")
    slurm_config = load_params(base_config["slurm_config"])

    command = " ".join(sys.argv).replace("--sbatch", "")
    cmd = 'sbatch %s --wrap="%s %s"' % (
        slurm_config[job_name],
        slurm_config["setup"],
        command,
    )
    print(cmd)
    # subprocess.run(cmd, shell=True, check=True, universal_newlines=True)


def build_clarg_params(
    args: argparse.Namespace, dannce_net: bool, prediction: bool
) -> Dict:
    """Build command line argument parameters

    Args:
        args (argparse.Namespace): Command line arguments parsed by argparse.
        dannce_net (bool): If true, use dannce net defaults.
        prediction (bool): If true, use prediction defaults.

    Returns:
        Dict: Parameters dictionary.
    """
    # Get the params specified in base config and io.yaml
    params = build_params(args.base_config, dannce_net)

    # Combine those params with the clargs
    params = combine(params, args, dannce_net)
    params = infer_params(params, dannce_net, prediction)
    check_config(params, dannce_net, prediction)
    return params


def add_shared_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    # Parse shared args for all conditions
    parser.add_argument(
        "base_config", metavar="base_config", help="Path to base config."
    )
    parser.add_argument("--sbatch", action="store_true", help="Run as sbatch job.")
    parser.add_argument("--viddir", dest="viddir", help="Directory containing videos.")
    parser.add_argument(
        "--crop-height",
        dest="crop_height",
        type=ast.literal_eval,
        help="Image crop height.",
    )
    parser.add_argument(
        "--crop-width",
        dest="crop_width",
        type=ast.literal_eval,
        help="Image crop width.",
    )
    parser.add_argument(
        "--camnames",
        dest="camnames",
        type=ast.literal_eval,
        help="List of ordered camera names.",
    )

    parser.add_argument("--io-config", dest="io_config", help="Path to io.yaml file.")

    parser.add_argument(
        "--n-channels-out",
        dest="n_channels_out",
        type=int,
        help="Number of keypoints to output. For COM, this is typically 1, but can be equal to the number of points tracked to run in MULTI_MODE.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Number of images per batch.",
    )
    parser.add_argument(
        "--sigma",
        dest="sigma",
        type=int,
        help="Standard deviation of confidence maps.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        help="verbose=0 prints nothing to std out. verbose=1 prints training summary to std out.",
    )
    parser.add_argument("--net", dest="net", help="Network architecture. See nets.py")
    parser.add_argument(
        "--gpu-id", dest="gpu_id", help="String identifying GPU to use."
    )
    parser.add_argument("--immode", dest="immode", help="Data format for images.")

    parser.add_argument(
        "--mono",
        dest="mono",
        type=ast.literal_eval,
        help="If true, converts 3-channel video frames into mono grayscale using standard RGB->gray conversion formula (ref. scikit-image).",
    )

    parser.add_argument(
        "--mirror",
        dest="mirror",
        type=ast.literal_eval,
        help="If true, uses a single video file for multiple views.",
    )

    parser.add_argument(
        "--norm-method",
        dest="norm_method",
        help="Normalization method to use, can be 'batch', 'instance', or 'layer'.",
    )

    parser.add_argument(
        "--predict-labeled-only", dest="predict_labeled_only", action="store_true"
    )

    parser.add_argument(
        "--training-fraction",
        dest="training_fraction",
        type=float,
    )

    parser.add_argument(
        "--custom-model",
        dest="custom_model",
    )

    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        type=int,
    )

    return parser


def add_shared_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all train modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--exp",
        dest="exp",
        type=ast.literal_eval,
        help="List of experiment dictionaries for network training. See examples in io.yaml.",
    )
    parser.add_argument(
        "--loss",
        dest="loss",
        help="Loss function to use during training. See losses.py.",
    )
    # Taken from changes by robb
    parser.add_argument(
        "--huber-delta",
        dest="huber-delta",
        type=float,
        help="Delta Value if using huber loss",
    )
    parser.add_argument(
        "--epochs", dest="epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num-validation-per-exp",
        dest="num_validation_per_exp",
        type=int,
        help="Number of validation images to use per recording during training.",
    )
    parser.add_argument(
        "--num-train-per-exp",
        dest="num_train_per_exp",
        type=int,
        help="Number of training images to use per recording during training.",
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        type=ast.literal_eval,
        help="List of additional metrics to report. See losses.py",
    )

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")

    parser.add_argument(
        "--augment-hue",
        dest="augment_hue",
        type=ast.literal_eval,
        help="If True, randomly augment hue of each image in training set during training.",
    )
    parser.add_argument(
        "--augment-brightness",
        dest="augment_brightness",
        type=ast.literal_eval,
        help="If True, randomly augment brightness of each image in training set during training.",
    )

    parser.add_argument(
        "--augment-hue-val",
        dest="augment_hue_val",
        type=float,
        help="If hue augmentation is True, chooses random hue delta in [-augment_hue_val, augment_hue_val]. Range = [0,1].",
    )
    parser.add_argument(
        "--augment-brightness-val",
        dest="augment_bright_val",
        type=float,
        help="If brightness augmentation is True, chooses random brightness delta in [-augment_hue_val, augment_hue_val]. Range = [0,1].",
    )
    parser.add_argument(
        "--augment-rotation-val",
        dest="augment_rotation_val",
        type=int,
        help="If continuous rotation augmentation is True, chooses random rotation angle in degrees in [-augment_rotation_val, augment_rotation_val]",
    )
    parser.add_argument(
        "--data-split-seed",
        dest="data_split_seed",
        type=int,
        help="Integer seed for the random numebr generator controlling train/test data splits",
    )
    parser.add_argument(
        "--valid-exp",
        dest="valid_exp",
        type=ast.literal_eval,
        help="Pass a list of the expfile indices (0-indexed, starting from the top of your expdict) to be set aside for validation",
    )
    return parser


def add_shared_predict_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all predict modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--max-num-samples",
        dest="max_num_samples",
        type=int,
        help="Maximum number of samples to predict during COM or DANNCE prediction.",
    )
    parser.add_argument(
        "--start-batch",
        dest="start_batch",
        type=int,
        help="Starting batch number during dannce prediction.",
    )
    parser.add_argument(
        "--start-sample",
        dest="start_sample",
        type=int,
        help="Starting sample number during dannce prediction.",
    )
    parser.add_argument("--multigpu", dest="multigpu", action="store_true")

    return parser


def add_dannce_shared_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all dannce modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--net-type",
        dest="net_type",
        help="Net types can be:\n"
        "AVG: more precise spatial average DANNCE, can be harder to train\n"
        "MAX: DANNCE where joint locations are at the maximum of the 3D output distribution\n",
    )
    parser.add_argument(
        "--com-fromlabels",
        dest="com_fromlabels",
        help="If True, uses the average 3D label position as the 3D COM. Inaccurate for frames with few labeled landmarks.",
    )
    parser.add_argument(
        "--medfilt-window",
        dest="medfilt_window",
        type=int,
        help="Sets the size of an optional median filter used to smooth the COM trace before DANNCE training or prediction.",
    )
    parser.add_argument(
        "--com-file",
        dest="com_file",
        help="Path to com file to use during dannce prediction.",
    )
    parser.add_argument(
        "--new-last-kernel-size",
        dest="new_last_kernel_size",
        type=ast.literal_eval,
        help="List denoting last 3d kernel size. Ex: --new-last-kernel-size=[3,3,3]",
    )
    parser.add_argument(
        "--new-n-channels_out",
        dest="new_n_channels_out",
        type=int,
        help="When finetuning, this refers to the new number of predicted keypoints.",
    )
    parser.add_argument(
        "--n-layers-locked",
        dest="n_layers_locked",
        type=int,
        help="Number of layers from model input to freeze during finetuning.",
    )
    parser.add_argument(
        "--vmin",
        dest="vmin",
        type=int,
        help="Minimum range of 3D grid. (Units of distance)",
    )
    parser.add_argument(
        "--vmax",
        dest="vmax",
        type=int,
        help="Maximum range of 3D grid. (Units of distance)",
    )
    parser.add_argument(
        "--nvox",
        dest="nvox",
        type=int,
        help="Number of voxels to span each dimension of 3D grid.",
    )
    parser.add_argument(
        "--interp",
        dest="interp",
        help="Voxel interpolation for 3D grid. Linear or nearest.",
    )
    parser.add_argument(
        "--depth",
        dest="depth",
        type=ast.literal_eval,
        help="If True, will append depth information when sampling images. Particularly useful when using just 1 cameras.",
    )

    parser.add_argument(
        "--comthresh",
        dest="comthresh",
        help="COM finder output confidence scores less than this threshold will be discarded.",
    )
    parser.add_argument(
        "--weighted",
        dest="weighted",
        type=ast.literal_eval,
        help="If True, will weight the COM estimate in each camera by its confidence score",
    )
    parser.add_argument(
        "--com-method",
        dest="com_method",
        help="Method of combining 3D COMs across camera pairs. Options: 'median', 'mean'",
    )
    parser.add_argument(
        "--cthresh",
        dest="cthresh",
        type=int,
        help="If the 3D COM has a coordinate beyond this value (in mm), discard it as an error.",
    )
    parser.add_argument(
        "--channel-combo",
        dest="channel_combo",
        help="Dictates whether or not to randomly shuffle the camera order when processing volumes. Options: 'None', 'random'",
    )
    parser.add_argument(
        "--n-views",
        dest="n_views",
        type=int,
        help="Sets the absolute number of views (when using fewer than 6 views only)",
    )
    parser.add_argument(
        "--train-mode",
        dest="train_mode",
        help="Training modes can be:\n"
        "new: initializes and trains a network from scratch\n"
        "finetune: loads in pre-trained weights and fine-tuned from there\n"
        "continued: initializes a full model, including optimizer state, and continuous training from the last full model checkpoint",
    )
    parser.add_argument(
        "--dannce-finetune-weights",
        dest="dannce_finetune_weights",
        help="Path to weights of initial model for dannce fine tuning.",
    )

    parser.add_argument(
        "--use-silhouette",
        type=ast.literal_eval,
        dest="use_silhouette",
    )

    parser.add_argument(
        "--use-silhouette-in-volume",
        default=False,
        type=ast.literal_eval,
        dest="use_silhouette_in_volume",
    )

    parser.add_argument(
        "--soft-silhouette",
        default=False,
        type=ast.literal_eval,
        dest="soft_silhouette",
    )
    parser.add_argument("--dataset", default="label3d", dest="dataset")
    return parser


def add_dannce_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to dannce training.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--dannce-train-dir",
        dest="dannce_train_dir",
        help="Training directory for dannce network.",
    )
    parser.add_argument(
        "--rotate",
        dest="rotate",
        type=ast.literal_eval,
        help="If True, use rotation augmentation for dannce training.",
    )
    parser.add_argument(
        "--augment-continuous-rotation",
        dest="augment_continuous_rotation",
        type=ast.literal_eval,
        help="If True, rotate all images in each sample of the training set by a random value between [-5 and 5] degrees during training.",
    )
    parser.add_argument(
        "--mirror-augmentation",
        dest="mirror_augmentation",
        type=ast.literal_eval,
        help="If True, mirror the images in half of the samples of the training set.",
    )
    parser.add_argument(
        "--drop-landmark",
        dest="drop_landmark",
        type=ast.literal_eval,
        help="Pass a list of landmark indices to exclude these landmarks from training",
    )
    parser.add_argument(
        "--use-npy",
        dest="use_npy",
        type=ast.literal_eval,
        help="If True, loads training data from npy files",
    )
    parser.add_argument(
        "--rand-view-replace",
        dest="rand_view_replace",
        type=ast.literal_eval,
        help="If True, samples n_rand_views with replacement",
    )
    parser.add_argument(
        "--n-rand-views",
        dest="n_rand_views",
        type=ast.literal_eval,
        help="Number of views to sample from the full viewset during training",
    )
    parser.add_argument(
        "--multi-gpu-train",
        dest="multi_gpu_train",
        type=ast.literal_eval,
        help="If True, distribute training data across multiple GPUs for each batch",
    )
    parser.add_argument(
        "--heatmap-reg",
        dest="heatmap_reg",
        type=ast.literal_eval,
        help="If True, use heatmap regularization during training",
    )
    parser.add_argument(
        "--heatmap-reg-coeff",
        dest="heatmap_reg_coeff",
        type=float,
        help="Sets the weight on the heatmap regularization term in the objective function.",
    )
    parser.add_argument(
        "--save-pred-targets",
        dest="save_pred_targets",
        type=ast.literal_eval,
        help="If True, save predictions evaluated at checkpoints during training. Note that for large training datasets, this can cause memory issues.",
    )
    parser.add_argument(
        "--avg-max",
        dest="avg+max",
        type=float,
        help="Pass a floating point value here for DANNCE to enter AVG+MAX training mode, where the 3D maps are MAX-like regularized to be Gaussian. The avg+max value is used to weight the contribution of the MAX-like loss.",
    )

    parser.add_argument(
        "--silhouette-loss-weight",
        type=float,
        dest="silhouette_loss_weight",
    )
    parser.add_argument(
        "--temporal-loss-weight",
        type=float,
        dest="temporal_loss_weight",
    )
    parser.add_argument(
        "--temporal-chunk-size",
        type=int,
        dest="temporal_chunk_size",
    )
    parser.add_argument(
        "--unlabeled-temp",
        type=float,
        dest="unlabeled_temp",
    )
    parser.add_argument("--support-exp", type=ast.literal_eval, dest="support_exp")
    parser.add_argument(
        "--n-support-chunks",
        type=int,
        dest="n_support_chunks",
    )

    parser.add_argument(
        "--separation-loss-weight",
        type=float,
        dest="separation_loss_weight",
    )

    parser.add_argument(
        "--separation-delta",
        type=float,
        dest="separation_delta",
    )

    # parser.add_argument(
    #     "--lr-scheduler",
    #     type=str,
    #     dest="lr_scheduler",
    # )

    parser.add_argument(
        "--symmetry-loss-weight",
        type=float,
        dest="symmetry_loss_weight",
    )

    return parser


def add_dannce_predict_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to dannce prediction.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--dannce-predict-dir",
        dest="dannce_predict_dir",
        help="Prediction directory for dannce network.",
    )
    parser.add_argument(
        "--dannce-predict-model",
        dest="dannce_predict_model",
        help="Path to model to use for dannce prediction.",
    )
    parser.add_argument(
        "--predict-model",
        dest="predict_model",
        help="Path to model to use for dannce prediction.",
    )
    parser.add_argument(
        "--expval",
        dest="expval",
        type=ast.literal_eval,
        help="If True, use expected value network. This is normally inferred from the network name. But because prediction can be decoupled from the net param, expval can be set independently if desired.",
    )
    parser.add_argument(
        "--from-weights",
        dest="from_weights",
        type=ast.literal_eval,
        help="If True, attempt to load in a prediction model without requiring a full model file (i.e. just using weights). May fail for some model types.",
    )
    parser.add_argument(
        "--write-npy",
        dest="write_npy",
        help="If not None, uses this base path to write large dataset to npy files",
    )
    parser.add_argument(
        "--label3d-index",
        dest="label3d_index",
        type=int,
        default=0,
    )

    return parser


def add_com_train_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to COM training.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--com-train-dir",
        dest="com_train_dir",
        help="Training directory for COM network.",
    )
    parser.add_argument(
        "--com-finetune-weights",
        dest="com_finetune_weights",
        help="Initial weights to use for COM finetuning.",
    )
    parser.add_argument(
        "--augment-shift",
        dest="augment_shift",
        type=ast.literal_eval,
        help="If True, shift all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-zoom",
        dest="augment_zoom",
        type=ast.literal_eval,
        help="If True, zoom all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-shear",
        dest="augment_shear",
        type=ast.literal_eval,
        help="If True, shear all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-rotation",
        dest="augment_rotation",
        type=ast.literal_eval,
        help="If True, rotate all images in each sample of the training set by a random value during training.",
    )
    parser.add_argument(
        "--augment-shear-val",
        dest="augment_shear_val",
        type=int,
        help="If shear augmentation is True, chooses random shear angle in degrees in [-augment_shear_val, augment_shear_val]",
    )
    parser.add_argument(
        "--augment-zoom-val",
        dest="augment_zoom_val",
        type=float,
        help="If zoom augmentation is True, chooses random zoom factor in [1-augment_zoom_val, 1+augment_zoom_val]",
    )
    parser.add_argument(
        "--augment-shift-val",
        dest="augment_shift_val",
        type=float,
        help="If shift augmentation is True, chooses random offset for rows and columns in [im_size*augment_shift_val, im_size*augment_shift_val]. So augment_shift_val is a fraction of the image size (must be in range [0,1])",
    )
    return parser


def add_com_predict_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments specific to COM prediction.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--com-predict-dir",
        dest="com_predict_dir",
        help="Prediction directory for COM network.",
    )
    parser.add_argument(
        "--com-predict-weights",
        dest="com_predict_weights",
        help="Path to .pth weights to use for COM prediction.",
    )
    return parser


def add_com_shared_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add arguments shared by all COM modes.

    Args:
        parser (argparse.ArgumentParser): Command line argument parser.

    Returns:
        argparse.ArgumentParser: Parser with added arguments.
    """
    parser.add_argument(
        "--dsmode",
        dest="dsmode",
        help="Downsampling mode. Can be dsm (local average) or nn (nearest_neighbor).",
    )
    parser.add_argument(
        "--downfac", dest="downfac", type=int, help="Downfactoring rate of images."
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        type=ast.literal_eval,
        help="If True, perform debugging operations.",
    )
    return parser


def add_multi_gpu_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("config", help="Path to .yaml configuration file")
    parser.add_argument(
        "--n-samples-per-gpu",
        dest="n_samples_per_gpu",
        type=int,
        default=5000,
        help="Number of samples for each GPU job to handle.",
    )
    parser.add_argument(
        "--only-unfinished",
        dest="only_unfinished",
        type=ast.literal_eval,
        default=False,
        help="If true, only predict chunks that have not been previously predicted.",
    )
    parser.add_argument(
        "--predict-path",
        dest="predict_path",
        default=None,
        help="When using only_unfinished, check predict_path for previously predicted chunks.",
    )
    parser.add_argument(
        "--com-file",
        dest="com_file",
        default=None,
        help="Use com-file to check the number of samples over which to predict rather than a dannce.mat file",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        type=ast.literal_eval,
        default=True,
        help="If True, print out submission command and info.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        type=ast.literal_eval,
        default=False,
        help="If True, print out submission command and info, but do not submit jobs.",
    )
    parser.add_argument(
        "--dannce-file",
        dest="dannce_file",
        default=None,
        help="Path to dannce.mat file to use for determining n total samples.",
    )


def combine(base_params: Dict, clargs: argparse.Namespace, dannce_net: bool) -> Dict:
    """Combine command line, io, and base configurations.

    Args:
        base_params (Dict): Parameters dictionary.
        clargs (argparse.Namespace): Command line argument namespace.
        dannce_net (bool): Description

    Returns:
        Dict: Parameters dictionary.
    """
    if dannce_net:
        alldefaults = {**_param_defaults_shared, **_param_defaults_dannce}
    else:
        alldefaults = {**_param_defaults_shared, **_param_defaults_com}

    # Logic ---
    # load defaults from parser if they are not already in config
    # use parser argument if different from the default
    for k, v in clargs.__dict__.items():
        if k in alldefaults:
            if v != alldefaults[k] or k not in base_params:
                base_params[k] = v
        elif v is not None:
            base_params[k] = v

    for k, v in base_params.items():
        print("{} set to: {}".format(k, v))
    return base_params


def get_parser():
    # Create the main parser
    parser = argparse.ArgumentParser()

    # Create subparsers for train and predict
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train a network.")
    predict_parser = subparsers.add_parser("predict", help="Predict using a network.")
    predict_multi_parser = subparsers.add_parser(
        "predict-multi-gpu", help="Predict across multiple gpus using a network."
    )
    merge_parser = subparsers.add_parser(
        "merge", help="Merge predictions from multiple gpus."
    )

    # Create subparsers for COM and DANNCE modes for train
    train_subparsers = train_parser.add_subparsers(dest="mode")
    train_com_parser = train_subparsers.add_parser(
        "com", help="Train a center of mass network."
    )
    train_dannce_parser = train_subparsers.add_parser(
        "dannce", help="Train a DANNCE network."
    )
    train_sdannce_parser = train_subparsers.add_parser(
        "sdannce", help="Train a SDANNCE network."
    )

    # Create subparsers for COM and DANNCE modes for predict
    predict_subparsers = predict_parser.add_subparsers(dest="mode")
    predict_com_parser = predict_subparsers.add_parser(
        "com", help="Predict with a center of mass network."
    )
    predict_dannce_parser = predict_subparsers.add_parser(
        "dannce", help="Predict with a DANNCE network."
    )
    predict_sdannce_parser = predict_subparsers.add_parser(
        "sdannce", help="Predict with a SDANNCE network."
    )

    # Create subparsers for COM and DANNCE modes for predict-multi-gpu
    predict_multi_subparsers = predict_multi_parser.add_subparsers(dest="mode")
    predict_multi_com_parser = predict_multi_subparsers.add_parser(
        "com", help="Predict with a center of mass network across multiple gpus."
    )
    predict_multi_dannce_parser = predict_multi_subparsers.add_parser(
        "dannce", help="Predict with a DANNCE network across multiple gpus."
    )
    predict_multi_sdannce_parser = predict_multi_subparsers.add_parser(
        "sdannce", help="Predict with a SDANNCE network across multiple gpus."
    )

    # Create subparsers for COM and DANNCE modes for merge
    merge_subparsers = merge_parser.add_subparsers(dest="mode")
    merge_com_parser = merge_subparsers.add_parser("com", help="Merge COM predictions.")
    merge_dannce_parser = merge_subparsers.add_parser(
        "dannce", help="Merge DANNCE predictions."
    )
    merge_sdannce_parser = merge_subparsers.add_parser(
        "sdannce", help="Merge SDANNCE predictions."
    )

    com_defaults = {**_param_defaults_shared, **_param_defaults_com}
    dannce_defaults = {**_param_defaults_shared, **_param_defaults_dannce}
    sdannce_defaults = {**_param_defaults_shared, **_param_defaults_dannce}
    # Set default values for each mode
    train_com_parser.set_defaults(**com_defaults)
    train_dannce_parser.set_defaults(**dannce_defaults)
    train_sdannce_parser.set_defaults(**sdannce_defaults)
    predict_com_parser.set_defaults(**com_defaults)
    predict_dannce_parser.set_defaults(**dannce_defaults)
    predict_sdannce_parser.set_defaults(**sdannce_defaults)

    # Add arguments for each subparser
    train_com_parser = add_shared_args(train_com_parser)
    train_com_parser = add_shared_train_args(train_com_parser)
    train_com_parser = add_com_shared_args(train_com_parser)
    train_com_parser = add_com_train_args(train_com_parser)

    train_dannce_parser = add_shared_args(train_dannce_parser)
    train_dannce_parser = add_shared_train_args(train_dannce_parser)
    train_dannce_parser = add_dannce_shared_args(train_dannce_parser)
    train_dannce_parser = add_dannce_train_args(train_dannce_parser)

    train_sdannce_parser = add_shared_args(train_sdannce_parser)
    train_sdannce_parser = add_shared_train_args(train_sdannce_parser)
    train_sdannce_parser = add_dannce_shared_args(train_sdannce_parser)
    train_sdannce_parser = add_dannce_train_args(train_sdannce_parser)

    predict_com_parser = add_shared_args(predict_com_parser)
    predict_com_parser = add_shared_predict_args(predict_com_parser)
    predict_com_parser = add_com_shared_args(predict_com_parser)
    predict_com_parser = add_com_predict_args(predict_com_parser)

    predict_dannce_parser = add_shared_args(predict_dannce_parser)
    predict_dannce_parser = add_shared_predict_args(predict_dannce_parser)
    predict_dannce_parser = add_dannce_shared_args(predict_dannce_parser)
    predict_dannce_parser = add_dannce_predict_args(predict_dannce_parser)

    predict_sdannce_parser = add_shared_args(predict_sdannce_parser)
    predict_sdannce_parser = add_shared_predict_args(predict_sdannce_parser)
    predict_sdannce_parser = add_dannce_shared_args(predict_sdannce_parser)
    predict_sdannce_parser = add_dannce_predict_args(predict_sdannce_parser)

    predict_multi_com_parser = add_multi_gpu_args(predict_multi_com_parser)
    predict_multi_dannce_parser = add_multi_gpu_args(predict_multi_dannce_parser)
    predict_multi_sdannce_parser = add_multi_gpu_args(predict_multi_sdannce_parser)
    merge_com_parser = add_multi_gpu_args(merge_com_parser)
    merge_dannce_parser = add_multi_gpu_args(merge_dannce_parser)
    merge_sdannce_parser = add_multi_gpu_args(merge_sdannce_parser)

    # Parse the arguments
    return parser.parse_args()


def main():
    """Entry point for the command line interface."""
    args = get_parser()

    # Handle slurm submission
    if args.sbatch and args.command != "predict-multi-gpu":
        submit_job(args)
        return

    # Handle slurm submission for multi-gpu prediction
    if args.command == "predict-multi-gpu":
        mgpu_args = args.__dict__.copy()
        del mgpu_args["command"]
        del mgpu_args["mode"]
        handler = MultiGpuHandler(**mgpu_args)
        if args.mode == "dannce":
            handler.submit_dannce_predict_multi_gpu()
        elif args.mode == "sdannce":
            handler.submit_sdannce_predict_multi_gpu()
        elif args.mode == "com":
            handler.submit_com_predict_multi_gpu()
        return

    # Handle merging of multi-gpu predictions
    if args.command == "merge":
        mgpu_args = args.__dict__.copy()
        del mgpu_args["command"]
        del mgpu_args["mode"]
        handler = MultiGpuHandler(**mgpu_args)
        if args.mode == "dannce":
            handler.dannce_merge()
        elif args.mode == "sdannce":
            handler.dannce_merge()
        elif args.mode == "com":
            handler.com_merge()
        return

    # Handle running on the current process
    params = build_clarg_params(
        args, dannce_net=(args.mode == "dannce"), prediction=(args.command == "predict")
    )
    if args.comand == "train":
        if args.mode == "dannce":
            dannce_train(params)
        elif args.mode == "sdannce":
            sdannce_train(params)
        elif args.mode == "com":
            com_train(params)
        return
    elif args.command == "predict":
        if args.mode == "dannce":
            dannce_predict(params)
        elif args.mode == "sdannce":
            sdannce_predict(params)
        elif args.mode == "com":
            com_predict(params)
        return
