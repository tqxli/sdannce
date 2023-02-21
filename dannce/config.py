from copy import deepcopy
import numpy as np
import imageio
import os, shutil
import yaml
from typing import Dict, Text
import warnings

from dannce.engine.data import io
from dannce import (
    _param_defaults_dannce,
    _param_defaults_shared,
    _param_defaults_com,
)

_DEFAULT_VIDDIR = "videos"
_DEFAULT_VIDDIR_SIL = "videos_sil"
_DEFAULT_COMSTRING = "COM"
_DEFAULT_COMFILENAME = "com3d.mat"
_DEFAULT_SEG_MODEL = "../weights/maskrcnn.pth"


def grab_predict_label3d_file(defaultdir="", index=0):
    """
    Finds the paths to the training experiment yaml files.
    """
    def_ep = os.path.join(".", defaultdir)
    label3d_files = os.listdir(def_ep)
    label3d_files = [
        os.path.join(def_ep, f) for f in label3d_files if "dannce.mat" in f
    ]
    # label3d_files.sort()
    label3d_files = sorted(label3d_files)
    # breakpoint()
    if len(label3d_files) == 0:
        raise Exception("Did not find any *dannce.mat file in {}".format(def_ep))
    print("Using the following *dannce.mat files: {}".format(label3d_files[index]))
    return label3d_files[index]


def infer_params(params, dannce_net, prediction):
    """
    Some parameters that were previously specified in configs can just be inferred
        from others, thus relieving config bloat
    """
    # Grab the camnames from *dannce.mat if not in config
    if params["camnames"] is None:
        f = grab_predict_label3d_file()
        params["camnames"] = io.load_camnames(f)
        if params["camnames"] is None:
            raise Exception("No camnames in config or in *dannce.mat")

    # Infer vid_dir_flag and extension and n_channels_in and chunks
    # from the videos and video folder organization.
    # Look into the video directory / camnames[0]. Is there a video file?
    # If so, vid_dir_flag = True
    viddir = os.path.join(params["viddir"], params["camnames"][0])
    video_files = os.listdir(viddir)

    if any([".mp4" in file for file in video_files]) or any(
        [".avi" in file for file in video_files]
    ):

        print_and_set(params, "vid_dir_flag", True)
    else:
        print_and_set(params, "vid_dir_flag", False)
        viddir = os.path.join(viddir, video_files[0])
        video_files = os.listdir(viddir)

    extension = ".mp4" if any([".mp4" in file for file in video_files]) else ".avi"
    print_and_set(params, "extension", extension)
    video_files = [file for file in video_files if extension in file]

    # Use the camnames to find the chunks for each video
    chunks = {}
    for name in params["camnames"]:
        if params["vid_dir_flag"]:
            camdir = os.path.join(params["viddir"], name)
        else:
            camdir = os.path.join(params["viddir"], name)
            intermediate_folder = os.listdir(camdir)
            camdir = os.path.join(camdir, intermediate_folder[0])
        video_files = os.listdir(camdir)
        video_files = [f for f in video_files if extension in f]
        video_files = sorted(video_files, key=lambda x: int(x.split(".")[0]))
        chunks[name] = np.sort([int(x.split(".")[0]) for x in video_files])

    print_and_set(params, "chunks", chunks)

    firstvid = str(chunks[params["camnames"][0]][0]) + params["extension"]
    camf = os.path.join(viddir, firstvid)

    # Infer n_channels_in from the video info
    v = imageio.get_reader(camf)
    im = v.get_data(0)
    v.close()
    print_and_set(params, "n_channels_in", im.shape[-1])

    # set the raw im height and width
    print_and_set(params, "raw_im_h", im.shape[0])
    print_and_set(params, "raw_im_w", im.shape[1])

    if dannce_net and params["avg+max"] is not None:
        # To use avg+max, need to start with an AVG network
        # In case the net type is not properly specified, set it here
        print_and_set(params, "expval", True)
        print_and_set(params, "net_type", "AVG")

    if dannce_net and params["net"] is None:
        # Here we assume that if the network and expval are specified by the user
        # then there is no reason to infer anything. net + expval compatibility
        # are subsequently verified during check_config()
        #
        # If both the net and expval are unspecified, then we use the simpler
        # 'net_type' + 'train_mode' to select the correct network and set expval.
        # During prediction, the train_mode might be missing, and in any case only the
        # expval needs to be set.
        if params["net_type"] is None:
            raise Exception("Without a net name, net_type must be specified")

        if not prediction and params["train_mode"] is None:
            raise Exception("Need to specific train_mode for DANNCE training")

    # print_and_set(params, "expval", True)
    if dannce_net:
        # infer crop_height and crop_width if None. Just use max dims of video, as
        # DANNCE does not need to crop.
        if params["crop_height"] is None or params["crop_width"] is None:
            im_h = []
            im_w = []
            for i in range(len(params["camnames"])):
                viddir = os.path.join(params["viddir"], params["camnames"][i])
                if not params["vid_dir_flag"]:
                    # add intermediate directory to path
                    viddir = os.path.join(
                        params["viddir"], params["camnames"][i], os.listdir(viddir)[0]
                    )
                video_files = sorted(os.listdir(viddir))
                camf = os.path.join(viddir, video_files[0])
                v = imageio.get_reader(camf)
                im = v.get_data(0)
                v.close()
                im_h.append(im.shape[0])
                im_w.append(im.shape[1])

            if params["crop_height"] is None:
                print_and_set(params, "crop_height", [0, np.max(im_h)])
            if params["crop_width"] is None:
                print_and_set(params, "crop_width", [0, np.max(im_w)])

        if params["max_num_samples"] is not None:
            if params["max_num_samples"] == "max":
                print_and_set(params, "maxbatch", "max")
            elif isinstance(params["max_num_samples"], (int, np.integer)):
                print_and_set(
                    params,
                    "maxbatch",
                    int(np.ceil(params["max_num_samples"] / params["batch_size"])),
                )
            else:
                raise TypeError("max_num_samples must be an int or 'max'")
        else:
            print_and_set(params, "maxbatch", "max")

        if params["start_sample"] is not None:
            if isinstance(params["start_sample"], (int, np.integer)):
                print_and_set(
                    params,
                    "start_batch",
                    int(params["start_sample"] // params["batch_size"]),
                )
            else:
                raise TypeError("start_sample must be an int.")
        else:
            print_and_set(params, "start_batch", 0)

        if params["vol_size"] is not None:
            print_and_set(params, "vmin", -1 * params["vol_size"] / 2)
            print_and_set(params, "vmax", params["vol_size"] / 2)

        if params["heatmap_reg"] and not params["expval"]:
            raise Exception(
                "Heatmap regularization enabled only for AVG networks -- you are using MAX"
            )

        if params["n_rand_views"] == "None":
            print_and_set(params, "n_rand_views", None)

    # There will be strange behavior if using a mirror acquisition system and are cropping images
    if params["mirror"] and params["crop_height"][-1] != params["raw_im_h"]:
        msg = "Note: You are using a mirror acquisition system with image cropping."
        msg = (
            msg
            + " All coordinates will be flipped relative to the raw image height, so ensure that your labels are also in that reference frame."
        )
        warnings.warn(msg)

    return params


def print_and_set(params, varname, value):
    # Should add new values to params in place, no need to return
    params[varname] = value
    print("Setting {} to {}.".format(varname, params[varname]))


def check_config(params, dannce_net, prediction):
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
                "{} not in parameters. Please add it, or use vol_size instead of vmin and vmax".format(
                    v
                )
            )


def check_camnames(camp):
    """
    Raises an exception if camera names contain '_'
    """
    if "camnames" in camp:
        for cam in camp["camnames"]:
            if "_" in cam:
                raise Exception("Camera names cannot contain '_' ")


def check_net_expval(params):
    """
    Raise an exception if the network and expval (i.e. AVG/MAX) are incompatible
    """
    if params["net"] is None:
        raise Exception("net is None. You must set either net or net_type.")
    if params["net_type"] is not None:
        if (
            params["net_type"] == "AVG"
            and "AVG" not in params["net"]
            and "expected" not in params["net"]
        ):
            raise Exception("net_type is set to AVG, but you are using a MAX network")
        if (
            params["net_type"] == "MAX"
            and "MAX" not in params["net"]
            and params["net"] != "unet3d_big"
        ):
            raise Exception("net_type is set to MAX, but you are using a AVG network")

    if (
        params["expval"]
        and "AVG" not in params["net"]
        and "expected" not in params["net"]
    ):
        raise Exception("expval is set to True but you are using a MAX network")
    if (
        not params["expval"]
        and "MAX" not in params["net"]
        and params["net"] != "unet3d_big"
    ):
        raise Exception("expval is set to False but you are using an AVG network")


def copy_config(results_dir, main_config, io_config):
    """
    Copies config files into the results directory, and creates results
        directory if necessary
    """
    print("Saving results to: {}".format(results_dir))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    mconfig = os.path.join(
        results_dir, "copy_main_config_" + main_config.split(os.sep)[-1]
    )
    dconfig = os.path.join(results_dir, "copy_io_config_" + io_config.split(os.sep)[-1])

    shutil.copyfile(main_config, mconfig)
    shutil.copyfile(io_config, dconfig)


def inherit_config(child, parent, keys):
    """
    If a key in keys does not exist in child, assigns the key-value in parent to
        child.
    """
    for key in keys:
        if key not in child.keys():
            child[key] = parent[key]
            print(
                "{} not found in io.yaml file, falling back to main config".format(key)
            )

    return child


def write_config(results_dir, configdict, message, filename="modelconfig.cfg"):
    """Write a dictionary of k-v pairs to file.

    A much more customizable configuration writer. Accepts a dictionary of
    key-value pairs and just writes them all to file,
    together with a custom message
    """
    f = open(results_dir + filename, "w")
    for key in configdict:
        f.write("{}: {}\n".format(key, configdict[key]))
    f.write("message:" + message)


def read_config(filename):
    """Read configuration file.

    :param filename: Path to configuration file.
    """
    with open(filename) as f:
        params = yaml.safe_load(f)

    return params


def make_paths_safe(params):
    """Given a parameter dictionary, loops through the keys and replaces any \\ or / with os.sep
    to promote OS agnosticism
    """
    for key in params.keys():
        if isinstance(params[key], str):
            params[key] = params[key].replace("/", os.sep)
            params[key] = params[key].replace("\\", os.sep)

    return params


def make_none_safe(pdict):
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


def check_unrecognized_params(params: Dict):
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


def build_params(base_config: Text, dannce_net: bool):
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


def adjust_loss_params(params):
    """
    Adjust parameters dictionary according to specific losses.

    Args:
        params (Dict): Parameters dictionary.

    Returns:
        Dict: Parameters dictionary.
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
            print("Batch size < temporal seq size; reducing temporal chunk size.")
            params["temporal_chunk_size"] = params["batch_size"]
            params["loss"]["TemporalLoss"]["temporal_chunk_size"] = params["batch_size"]

    # option for using downsampled temporal sequences
    try:
        downsample = params["loss"]["TemporalLoss"]["downsample"]
    except:
        downsample = 1
    params["downsample"] = downsample

    if "PairRepulsionLoss" in params["loss"]:
        params["social_training"] = True

    if "ConsistencyLoss" in params["loss"]:
        # number of copies per unique training sample
        copies_per_sample = params["loss"]["ConsistencyLoss"].get(
            "copies_per_sample", 1
        )
        params["loss"]["ConsistencyLoss"]["copies_per_sample"] = copies_per_sample
        # do not exceed the specified batch size to avoid OOM
        params["form_batch"] = True
        n_samples_unique = params["batch_size"] // copies_per_sample
        params["form_bs"] = params["batch_size"]
        # adjust batch size to the number of unique samples
        # populate with augmented samples on the fly during training
        params["batch_size"] = n_samples_unique

    return params


def setup_train(params):
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
        print(
            "Using default n_rand_views augmentation with {} views and with replacement".format(
                params["n_views"]
            )
        )
        print("To disable n_rand_views augmentation, set it to None in the config.")
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


def setup_predict(params):
    # Depth disabled until next release.
    params["depth"] = False
    # Make the prediction directory if it does not exist.

    params["net_name"] = params["net"]
    params["n_views"] = int(params["n_views"])

    params["downsample"] = 1

    # While we can use experiment files for DANNCE training,
    # for prediction we use the base data files present in the main config
    # Grab the input file for prediction
    if params["dataset"] != "rat7m":
        params["label3d_file"] = grab_predict_label3d_file(
            index=params["label3d_index"]
        )
        params["base_exp_folder"] = os.path.dirname(params["label3d_file"])
    params["multi_mode"] = False

    print("Using camnames: {}".format(params["camnames"]))
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

    if params.get("n_instances") > 1:
        params["social_training"] = True
    if params["social_training"]:
        # load in parameters from the other animal
        comfile = params["com_file"]

        paired_expdict = {
            "label3d_file": params["label3d_file"],  # .replace('RAT1', 'RAT2'),
            "com_file": comfile,  # new_comfile,
        }
        paired_exp = deepcopy(params)
        for k, v in paired_expdict.items():
            paired_exp[k] = v
        params["experiment"][1] = paired_exp

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

    # if params.get("social_joint_training", False):
    #     params["social_training"] = False
    #     params["n_channels_out"] *= 2
    #     valid_params["n_channels_out"] *= 2
    #     params["n_markers"] *= 2

    return params, valid_params


def setup_com_train(params):
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


def setup_com_predict(params):
    params["multi_mode"] = MULTI_MODE = (
        params["n_channels_out"] > 1 & params["n_instances"] == 1
    )
    params["n_channels_out"] = params["n_channels_out"] + int(MULTI_MODE)

    # Grab the input file for prediction
    params["label3d_file"] = grab_predict_label3d_file(index=params["label3d_index"])

    print("Using camnames: {}".format(params["camnames"]))

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
