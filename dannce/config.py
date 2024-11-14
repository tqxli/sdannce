import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Union
import itertools

import imageio
import numpy as np
import yaml
from loguru import logger

from dannce import (
    _param_defaults_com,
    _param_defaults_dannce,
    _param_defaults_shared,
)
from dannce.engine.data import io

# imported from other files
_DEFAULT_VIDDIR = "videos"
_DEFAULT_VIDDIR_SIL = "videos_sil"
_DEFAULT_COMSTRING = "COM"
_DEFAULT_COMFILENAME = "com3d.mat"
_DEFAULT_SEG_MODEL = "../weights/maskrcnn.pth"


def grab_predict_label3d_file(default_dir: str = "", index: int = 0) -> str:
    """
    Finds the paths to the training experiment yaml files.

    Returns a string filename.
    """
    default_exp_path = Path(".", default_dir)
    label3d_files = list(default_exp_path.glob("*dannce.mat"))
    label3d_files = sorted(label3d_files)

    if not label3d_files:
        raise Exception(f"Did not find any *dannce.mat file in {default_exp_path}")
    logger.info(f"Using the following *dannce.mat files: {label3d_files[index]}")
    return str(label3d_files[index])


def infer_params(params: dict, dannce_net: bool, prediction: bool) -> dict:
    """
    Infer parameters not explicitly specified in the configs from other parameters + context.

    Args:
        dannce_net (bool): True if DANNCE else COM.
        prediction: True if running prediction/inference; False if training model

    Infer the following parameters (might be skipped in some cases)
    1. camnames
    2. vid_dir_flag
    3. extension
    4. chunks
    5. n_channels_in
    6. raw_im_h
    7. raw_im_w
    8. crop_height
    9. crop_width
    10. maxbatch
    11. start_batch
    12. vmin
    13. vmax
    14. n_rand_views
    """

    # 1. camnames
    ################################
    # Grab the camnames from *dannce.mat if not in config
    if params["camnames"] is None:
        label3d_filename = grab_predict_label3d_file()
        _camnames = io.load_camnames(label3d_filename)
        if _camnames is None:
            raise Exception("No camnames in config or in *dannce.mat")
        print_and_set(params, "camnames", _camnames)

    # base directory containing videos
    # picked from either: current directory; "exp" list; or "com_exp" list
    #   depending on COM/DANNCE and Train/Predict mode.
    example_base_dir = get_base_dir(params, dannce_net, prediction)
    logger.info(f"Using recording folder to infer video parameters: {example_base_dir}")

    # 2: vid_dir_flag
    ################################
    # check if the first camera folder contains a valid video file (mp4 or avi)
    # then set vid_dir_flag True (meaning viddir directly contains video files)
    cam1_dir = Path(example_base_dir, params["viddir"], params["camnames"][0])
    first_video_file = get_first_video_file(cam1_dir)
    if first_video_file is not None:
        print_and_set(params, "vid_dir_flag", True)
    else:
        try:
            # look for a subfolder containing video files
            inner_dir = next(cam1_dir.glob("*/"))
        except StopIteration:
            raise Exception(
                f"No .mp4 or .avi file found in viddir and viddir does not contain an inner directory. Checked dir: {cam1_dir}"
            )
        first_video_file = get_first_video_file(inner_dir)
        print_and_set(params, "vid_dir_flag", False)

    # 3: extension
    ################################
    if first_video_file.suffix == ".mp4":
        print_and_set(params, "extension", ".mp4")
    elif first_video_file.suffix == ".avi":
        print_and_set(params, "extension", ".avi")
    else:
        raise Exception(f"Invalid file extension: {first_video_file}")

    # 4: chunks
    ################################
    # Use the camnames to find the chunks for each video
    chunks = {}
    for camname in params["camnames"]:
        camdir = Path(example_base_dir, params["viddir"], camname)
        if not params["vid_dir_flag"]:
            # first folder in camera folder
            camdir = next(camdir.glob("*/"))
        video_files = list(camdir.glob("*" + params["extension"]))
        video_files = sorted(video_files, key=lambda vf: int(vf.stem))
        chunks[camname] = np.sort([int(vf.stem) for vf in video_files])

    print_and_set(params, "chunks", chunks)

    # 5,6,7: n_channels_in, raw_im_h, raw_im_w
    ###########################################
    # read first frame of video to get metadata
    # only infer if these values are unset
    if (
        params["n_channels_in"] is None
        or params["raw_im_h"] is None
        or params["raw_im_w"] is None
    ):
        v = imageio.get_reader(first_video_file)
        im = v.get_data(0)
        v.close()
        print_and_set(params, "n_channels_in", im.shape[-1])
        print_and_set(params, "raw_im_h", im.shape[0])
        print_and_set(params, "raw_im_w", im.shape[1])

    # Check dannce_type and "net" validity
    ######################################
    if dannce_net and params["net"] is None:
        # Here we assume that if the network and expval are specified by the user
        # then there is no reason to infer anything. net + expval compatibility
        # are subsequently verified during check_config()

        # If both the net and expval are unspecified, then we use the simpler
        # 'net_type' + 'train_mode' to select the correct network and set expval.
        # During prediction, the train_mode might be missing, and in any case only the
        # expval needs to be set.
        if params["net_type"] is None:
            raise Exception("Without a net name, net_type must be specified")

        if not prediction and params["train_mode"] is None:
            raise Exception("Need to specific train_mode for DANNCE training")

    # ignore for COMs
    if dannce_net:
        # 8,9: crop_height, crop_width
        ###########################################
        # DANNCE does not need to crop like COM net, so we can use the full video dimension
        if params["crop_height"] is None or params["crop_width"] is None:
            max_h = -1
            max_w = -1
            for camname in params["camnames"]:
                viddir = Path(example_base_dir, params["viddir", camname])
                if not params["vid_dir_flag"]:
                    # set viddir to inner folder
                    viddir = next(viddir.glob("*/"))

                video_files = sorted(os.listdir(viddir))
                camf = video_files[0]
                v = imageio.get_reader(camf)
                im = v.get_data(0)
                v.close()
                this_h = im.shape[0]
                this_w = im.shape[1]
                max_h = max(this_h, max_h)
                max_w = max(this_w, max_w)

            if params["crop_height"] is None:
                print_and_set(params, "crop_height", [0, max_h])
            if params["crop_width"] is None:
                print_and_set(params, "crop_width", [0, max_w])

        # 10: maxbatch
        ###########################################
        if params["max_num_samples"] == "max" or params["max_num_samples"] is None:
            print_and_set(params, "maxbatch", "max")
        elif isinstance(params["max_num_samples"], (int, np.integer)):
            maxbatch = int(np.ceil(params["max_num_samples"] / params["batch_size"]))
            print_and_set(params, "maxbatch", maxbatch)
        else:
            raise TypeError("max_num_samples must be an int or 'max'")

        # 11: start_batch
        ###########################################
        if params["start_sample"] is None:
            print_and_set(params, "start_batch", 0)
        elif isinstance(params["start_sample"], (int, np.integer)):
            start_batch = int(params["start_sample"] // params["batch_size"])
            print_and_set(params, "start_batch", start_batch)
        else:
            raise TypeError("start_sample must be an int.")

        # 12,13: vmin,vmax
        ###########################################
        if params["vol_size"] is not None:
            print_and_set(params, "vmin", -1 * params["vol_size"] / 2)
            print_and_set(params, "vmax", params["vol_size"] / 2)

        # verify heatmap regeulariziation
        ###########################################
        if params["heatmap_reg"] and not params["expval"]:
            raise Exception(
                "Heatmap regularization enabled only for AVG networks -- you are using MAX"
            )

        # 14: n_rand_views
        ###########################################
        if params["n_rand_views"] == "None":
            print_and_set(params, "n_rand_views", None)

    ##################################
    # There will be strange behavior if using a mirror acquisition system and are cropping images
    if params["mirror"] and params["crop_height"][-1] != params["raw_im_h"]:
        msg = "Note: You are using a mirror acquisition system with image cropping."
        msg = (
            msg
            + " All coordinates will be flipped relative to the raw image height, so ensure that your labels are also in that reference frame."
        )
        warnings.warn(msg)

    return params


def print_and_set(params: dict, varname: str, value: any):
    """Updates params dict and logs the value"""
    # Should add new values to params in place, no need to return
    params[varname] = value
    logger.warning(f"Setting {varname} to {params[varname]}.")


def check_config(params: dict, dannce_net: bool, prediction: bool):
    """
    Add parameter checks and restrictions here.
    """
    check_camnames(params)

    if params["exp"] is not None:
        for expdict in params["exp"]:
            check_camnames(expdict)

    if dannce_net:
        # check_net_expval(params)
        check_vmin_vmax(params)


def check_vmin_vmax(params):
    for v in ["vmin", "vmax", "nvox"]:
        if params[v] is None:
            raise Exception(
                f"{v} not in parameters. Please add it, or use vol_size instead of vmin and vmax"
            )


def check_camnames(camp):
    """
    Raises an exception if camera names contain '_'
    """
    if "camnames" in camp:
        for cam in camp["camnames"]:
            if "_" in cam:
                raise Exception("Camera names cannot contain '_' ")


# def copy_config(results_dir, main_config, io_config):
#     """
#     Copies config files into the results directory, and creates results
#         directory if necessary
#     """
#     print("Saving results to: {}".format(results_dir))

#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)

#     mconfig = os.path.join(
#         results_dir, "copy_main_config_" + main_config.split(os.sep)[-1]
#     )
#     dconfig = os.path.join(results_dir, "copy_io_config_" + io_config.split(os.sep)[-1])

#     shutil.copyfile(main_config, mconfig)
#     shutil.copyfile(io_config, dconfig)


def inherit_config(child: dict, parent: dict, keys: list):
    """
    If a key in keys does not exist in child, assigns the key-value in parent to
        child.
    """
    for key in keys:
        if key not in child.keys():
            child[key] = parent[key]
            logger.warning(
                "{} not found in io.yaml file, falling back to default".format(key)
            )

    return child


def write_config(
    results_dir: str, configdict: dict, message: str, filename="modelconfig.cfg"
):
    """Write a dictionary of k-v pairs to file.

    A much more customizable configuration writer. Accepts a dictionary of
    key-value pairs and just writes them all to file,
    together with a custom message
    """
    f = open(results_dir + filename, "w")
    for key in configdict:
        f.write("{}: {}\n".format(key, configdict[key]))
    f.write("message:" + message)


def read_config(filename: str):
    """Read configuration file.

    :param filename: Path to configuration file.
    """
    with open(filename) as f:
        params = yaml.safe_load(f)

    return params


def make_paths_safe(params: dict):
    """Given a parameter dictionary, loops through the keys and replaces any \\ or / with os.sep
    to promote OS agnosticism
    """
    for key in params.keys():
        if isinstance(params[key], str):
            params[key] = params[key].replace("/", os.sep)
            params[key] = params[key].replace("\\", os.sep)

    return params


def make_none_safe(pdict: dict):
    if isinstance(pdict, dict):
        for key in pdict:
            pdict[key] = make_none_safe(pdict[key])
    else:
        if (
            pdict is None
            or (isinstance(pdict, list) and None in pdict)
            or (isinstance(pdict, tuple) and None in pdict)
        ):
            return "None"
        else:
            return pdict
    return pdict


def check_unrecognized_params(params: dict):
    """Check for invalid keys in the params dict against param defaults.

    Args:
        params (Dict): Parameters dictionary.

    Raises:
        ValueError: Error if there are unrecognized keys in the configs.
    """
    # Check if key in any of the defaults
    invalid_keys = []
    for key in params:
        in_com = key in _param_defaults_com
        in_dannce = key in _param_defaults_dannce
        in_shared = key in _param_defaults_shared
        if not (in_com or in_dannce or in_shared):
            invalid_keys.append(key)

    # If there are any keys that are invalid, throw an error and print them out
    if len(invalid_keys) > 0:
        invalid_key_msg = [" %s," % key for key in invalid_keys]
        msg = "Unrecognized keys in the configs: %s" % "".join(invalid_key_msg)
        raise ValueError(msg)


def build_params(base_config: str, dannce_net: bool):
    """Build parameters dictionary from base config and io.yaml

    Args:
        base_config (Text): Path to base configuration .yaml.
        dannce_net (bool): If True, use dannce net defaults.

    Returns:
        Dict: Parameters dictionary.
    """
    base_params = read_config(base_config)
    base_params = make_paths_safe(base_params)
    params = read_config(base_params["io_config"])
    params = make_paths_safe(params)
    params = inherit_config(params, base_params, list(base_params.keys()))
    check_unrecognized_params(params)
    return params


def adjust_loss_params(params: dict):
    """
    Adjust parameters dictionary according to specific losses.

    Args:
        params (dict): Parameters dictionary.

    Returns:
        dict: Parameters dictionary.
    """

    # turn on flags for losses that require changes in inputs
    if params["use_silhouette_in_volume"]:
        params["use_silhouette"] = True
        params["n_rand_views"] = None

    if "SilhouetteLoss" in params["loss"] or "SilhouetteLoss2D" in params["loss"]:
        params["use_silhouette"] = True

    if "TemporalLoss" in params["loss"]:
        params["use_temporal"] = True
        params["temporal_chunk_size"] = temp_n = params["loss"]["TemporalLoss"][
            "temporal_chunk_size"
        ]

        # by default, the maximum batch size should be >= temporal seq len
        if params["batch_size"] < temp_n:
            logger.warning(
                "Batch size < temporal seq size; reducing temporal chunk size."
            )
            params["temporal_chunk_size"] = params["batch_size"]
            params["loss"]["TemporalLoss"]["temporal_chunk_size"] = params["batch_size"]

    # option for using downsampled temporal sequences
    try:
        downsample = params["loss"]["TemporalLoss"]["downsample"]
    except:
        downsample = 1
    params["downsample"] = downsample

    if "PairRepulsionLoss" in params["loss"]:
        params["is_social_dataset"] = True

    if "ConsistencyLoss" in params["loss"]:
        # number of copies per unique training sample
        copies_per_sample = params["loss"]["ConsistencyLoss"].get(
            "copies_per_sample", 1
        )
        params["loss"]["ConsistencyLoss"]["copies_per_sample"] = copies_per_sample
        # do not exceed the specified batch size to avoid OOM
        params["batch_augmentation"] = True
        n_samples_unique = params["batch_size"] // copies_per_sample
        params["batch_aug_size"] = params["batch_size"]
        # adjust batch size to the number of unique samples
        # populate with augmented samples on the fly during training
        params["batch_size"] = n_samples_unique

    if "BoneLengthLoss" in params["loss"]:
        params["loss"]["BoneLengthLoss"]["body_profile"] = params.get(
            "skeleton", "rat23"
        )

    return params


def setup_train(params: dict):
    # turn off currently unavailable features
    params["multi_mode"] = False
    params["depth"] = False

    # Default to 6 views but a smaller number of views can be specified in the
    # DANNCE config. If the legnth of the camera files list is smaller than
    # n_views, relevant lists will be duplicated in order to match n_views, if
    # possible.
    params["n_views"] = int(params["n_views"])
    if params["dataset"] == "rat7m":
        params["n_channels_out"] = 20
    elif params["dataset"] == "pair":
        params["n_channels_out"] = 12

    params = adjust_loss_params(params)

    # generator params
    cam3_train = True if params["cam3_train"] else False
    # We apply data augmentation with another data generator class
    randflag = params["channel_combo"] == "random"
    outmode = "coordinates" if params["expval"] else "3dprob"

    if params["use_npy"]:
        # mono conversion will happen from RGB npy files, and the generator
        # needs to b aware that the npy files contain RGB content
        params["chan_num"] = params["n_channels_in"]
    else:
        # Used to initialize arrays for mono, and also in *frommem (the final generator)
        params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    if cam3_train:
        params["n_rand_views"] = 3
        params["rand_view_replace"] = False
        randflag = True

    if params["n_rand_views"] == 0:
        logger.info(
            "Using default n_rand_views augmentation with {} views and with replacement".format(
                params["n_views"]
            )
        )
        logger.warning(
            "To disable n_rand_views augmentation, set it to None in the config."
        )
        params["n_rand_views"] = params["n_views"]
        params["rand_view_replace"] = True

    base_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["new_n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "depth": params["depth"],
        "mode": outmode,
        # "camnames": camnames,
        "immode": params["immode"],
        "shuffle": False,  # will shuffle later
        "rotation": False,  # will rotate later if desired
        # "vidreaders": vids,
        "distort": True,
        "crop_im": False,
        # "chunks": total_chunks,
        "mono": params["mono"],
        "mirror": params["mirror"],
    }

    # dataset params
    shared_args = {
        "chan_num": params["chan_num"],
        "expval": params["expval"],
        "nvox": params["nvox"],
        "heatmap_reg": params["heatmap_reg"],
        "heatmap_reg_coeff": params["heatmap_reg_coeff"],
        "occlusion": params["downscale_occluded_view"],
    }

    shared_args_train = {
        "rotation": params["rotate"],
        "augment_hue": params["augment_hue"],
        "augment_brightness": params["augment_brightness"],
        "augment_continuous_rotation": params["augment_continuous_rotation"],
        "mirror_augmentation": params["mirror_augmentation"],
        "right_keypoints": params["right_keypoints"],
        "left_keypoints": params["left_keypoints"],
        "bright_val": params["augment_bright_val"],
        "hue_val": params["augment_hue_val"],
        "rotation_val": params["augment_rotation_val"],
        "replace": params["rand_view_replace"],
        "random": randflag,
        "n_rand_views": params["n_rand_views"],
    }

    shared_args_valid = {
        "rotation": False,
        "augment_hue": False,
        "augment_brightness": False,
        "augment_continuous_rotation": False,
        "mirror_augmentation": False,
        "shuffle": False,
        "replace": params["allow_valid_replace"],
        # "replace": False,
        "n_rand_views": params["n_rand_views"]
        if params["allow_valid_replace"] or cam3_train
        else None,
        # "n_rand_views": params["n_rand_views"] if cam3_train else None,
        "random": True if cam3_train else False,
    }

    return params, base_params, shared_args, shared_args_train, shared_args_valid


def setup_predict(params: dict):
    # Depth disabled until next release.
    params["depth"] = False
    # Make the prediction directory if it does not exist.

    params["net_name"] = params["net"]
    params["n_views"] = int(params["n_views"])

    params["downsample"] = 1

    if "n_instances" not in params:
        params["n_instances"] = 1
    params["is_social_dataset"] = params["n_instances"] > 1

    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    if params["dataset"] != "rat7m":
        params["label3d_file"] = grab_predict_label3d_file(
            index=params["label3d_index"]
        )
        params["base_exp_folder"] = os.path.dirname(params["label3d_file"])
    params["multi_mode"] = False

    logger.info(f"Using camnames: {params['camnames']}")
    # Also add parent params under the 'experiment' key for compatibility
    # with DANNCE's video loading function
    if (params["use_silhouette_in_volume"]) or (
        params["write_visual_hull"] is not None
    ):
        params["viddir_sil"] = os.path.join(
            params["base_exp_folder"], _DEFAULT_VIDDIR_SIL
        )

    params["experiment"] = {}
    params["experiment"][0] = params

    if params["is_social_dataset"]:
        # repeat parameters for the remaining animals (besides instance_0)
        for i in range(1, params["n_instances"]):
            comfile = params["com_file"]

            paired_expdict = {
                "label3d_file": params["label3d_file"],
                "com_file": comfile,
            }
            paired_exp = deepcopy(params)
            for k, v in paired_expdict.items():
                paired_exp[k] = v
            params["experiment"][i] = paired_exp

    if params["start_batch"] is None:
        params["start_batch"] = 0
        params["save_tag"] = None
    else:
        params["save_tag"] = params["start_batch"]

    if params["new_n_channels_out"] is not None:
        params["n_markers"] = params["new_n_channels_out"]
    else:
        params["n_markers"] = params["n_channels_out"]

    # For real mono prediction
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]

    # generator params
    valid_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": params["batch_size"],
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "vmin": params["vmin"],
        "vmax": params["vmax"],
        "nvox": params["nvox"],
        "interp": params["interp"],
        "depth": params["depth"],
        "channel_combo": params["channel_combo"],
        "mode": "coordinates",
        # "camnames": camnames,
        "immode": params["immode"],
        "shuffle": False,
        "rotation": False,
        # "vidreaders": vids,
        "distort": True,
        "expval": params["expval"],
        "crop_im": False,
        # "chunks": params["chunks"],
        "mono": params["mono"],
        "mirror": params["mirror"],
        "predict_flag": True,
    }

    return params, valid_params


def setup_com_train(params: dict):
    # os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]

    # MULTI_MODE is where the full set of markers is trained on, rather than
    # the COM only. In some cases, this can help improve COMfinder performance.
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]
    params["multi_mode"] = (params["n_channels_out"] > 1) & (params["n_instances"] == 1)
    params["n_channels_out"] = params["n_channels_out"] + int(params["multi_mode"])

    params["lr"] = float(params["lr"])

    train_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        # "camnames": camnames,
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "downsample": params["downfac"],
        "shuffle": False,
        # "chunks": total_chunks,
        "dsmode": params["dsmode"],
        "mono": params["mono"],
        "mirror": params["mirror"],
    }

    valid_params = deepcopy(train_params)
    valid_params["shuffle"] = False

    return params, train_params, valid_params


def setup_com_predict(params: dict):
    params["multi_mode"] = MULTI_MODE = (params["n_channels_out"] > 1) and (
        params["n_instances"] == 1
    )
    params["n_channels_out"] = params["n_channels_out"] + int(MULTI_MODE)

    # Grab the input file for prediction
    params["label3d_file"] = grab_predict_label3d_file(index=params["label3d_index"])

    logger.info(f"Using camnames: {params['camnames']}")

    params["experiment"] = {}
    params["experiment"][0] = params

    # For real mono training
    params["chan_num"] = 1 if params["mono"] else params["n_channels_in"]
    dh = (params["crop_height"][1] - params["crop_height"][0]) // params["downfac"]
    dw = (params["crop_width"][1] - params["crop_width"][0]) // params["downfac"]
    params["input_shape"] = (dh, dw)

    if params["com_predict_weights"] is None:
        wdir = params["com_train_dir"]
        weights = os.listdir(wdir)
        weights = [f for f in weights if (".pth" in f) and ("epoch" in f)]
        weights = sorted(weights, key=lambda x: int(x.split(".")[0].split("epoch")[-1]))
        weights = weights[-1]
        params["com_predict_weights"] = os.path.join(wdir, weights)

    params["lr"] = float(params["lr"])

    predict_params = {
        "dim_in": (
            params["crop_height"][1] - params["crop_height"][0],
            params["crop_width"][1] - params["crop_width"][0],
        ),
        "n_channels_in": params["n_channels_in"],
        "batch_size": 1,
        "n_channels_out": params["n_channels_out"],
        "out_scale": params["sigma"],
        "camnames": {0: params["camnames"]},
        "crop_width": params["crop_width"],
        "crop_height": params["crop_height"],
        "downsample": params["downfac"],
        "labelmode": "coord",
        "chunks": params["chunks"],
        "shuffle": False,
        "dsmode": params["dsmode"],
        "mono": params["mono"],
        "mirror": params["mirror"],
        "predict_flag": True,
    }

    return params, predict_params


def get_first_video_file(p: Path) -> Union[Path, None]:
    """
    Given a folder, return a Path object of the first video file (avi or mp4) within.
    If muliple files, return the first video file sorted alphabetically.

    Otherwise return None.
    """
    video_files = list(itertools.chain(p.glob("*.mp4"), p.glob("*.avi")))
    video_files = sorted(video_files)
    if not video_files:
        return None
    return video_files[0]


def get_base_dir(params: dict, dannce_net: bool, prediction: bool) -> Path:
    """Get a base folder given the current settings

    For prediction:
        -> current directory -> videos 
    For (S)DANNCE training:
        -> exp[0].label3d_file > videos
    For COM training:
        -> com_exp[0].label3d_file > videos

    """
    if prediction:
        base_dir = Path.cwd()
    else:  # training network
        if dannce_net:  # (S)DANNCE network
            base_dir = Path(params["exp"][0]["label3d_file"]).parent
        else:  # COM network
            base_dir = Path(params["com_exp"][0]["label3d_file"]).parent

    return base_dir
