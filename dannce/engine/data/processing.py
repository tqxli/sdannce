"""Processing functions for dannce."""
import numpy as np
import imageio
import os
import PIL
from six.moves import cPickle
from typing import Dict, List, Text, Union
import pickle
import torch
from tqdm import tqdm
from copy import deepcopy
import yaml
from loguru import logger

import scipy.io as sio
from scipy.ndimage.filters import maximum_filter
from skimage import measure
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean as dsm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dannce.engine.data import serve_data_DANNCE, io
from dannce.config import make_paths_safe, make_none_safe
from dannce.config import _DEFAULT_VIDDIR, _DEFAULT_VIDDIR_SIL

"""
VIDEO
"""


def initialize_vids(params, datadict, e, vids, pathonly=True, vidkey="viddir"):
    """
    Initializes video path dictionaries for a training session. This is different
        than a predict session because it operates over a single animal ("experiment")
        at a time
    """
    for i in range(len(params["experiment"][e]["camnames"])):
        # Rather than opening all vids, only open what is needed based on the
        # maximum frame ID for this experiment and Camera
        flist = []
        for key in datadict.keys():
            if int(key.split("_")[0]) == e:
                flist.append(
                    datadict[key]["frames"][params["experiment"][e]["camnames"][i]]
                )

        flist = max(flist)

        # For COM prediction, we don't prepend experiment IDs
        # So detect this case and act accordingly.
        basecam = params["experiment"][e]["camnames"][i]
        if "_" in basecam:
            basecam = basecam.split("_")[1]

        if params["vid_dir_flag"]:
            addl = ""
        else:
            addl = os.listdir(
                os.path.join(
                    params["experiment"][e][vidkey],
                    basecam,
                    # "pandas"
                )
            )[0]
        r = generate_readers(
            params["experiment"][e][vidkey],
            os.path.join(basecam, addl),
            maxopt=flist,  # Large enough to encompass all videos in directory.
            extension=params["experiment"][e]["extension"],
            pathonly=pathonly,
        )

        if "_" in params["experiment"][e]["camnames"][i]:
            vids[params["experiment"][e]["camnames"][i]] = {}
            for key in r:
                vids[params["experiment"][e]["camnames"][i]][str(e) + "_" + key] = r[
                    key
                ]
        else:
            vids[params["experiment"][e]["camnames"][i]] = r

    return vids


def initialize_all_vids(params, datadict, exps, pathonly=True, vidkey="viddir"):
    vids = {}
    for e in exps:
        vids = initialize_vids(params, datadict, e, vids, pathonly, vidkey)
    return vids


def generate_readers(
    viddir, camname, minopt=0, maxopt=300000, pathonly=False, extension=".mp4"
):
    """Open all mp4 objects with imageio, and return them in a dictionary."""
    out = {}
    mp4files = [
        os.path.join(camname, f)
        for f in os.listdir(os.path.join(viddir, camname))
        if extension in f
        and (f[0] != "_")
        and (f[0] != ".")
        and int(f.rsplit(extension)[0]) <= maxopt
        and int(f.rsplit(extension)[0]) >= minopt
    ]
    # This is a trick (that should work) for getting rid of
    # awkward sub-directory folder names when they are being used
    mp4files_scrub = [
        os.path.join(
            os.path.normpath(f).split(os.sep)[0], os.path.normpath(f).split(os.sep)[-1]
        )
        for f in mp4files
    ]

    pixelformat = "yuv420p"
    input_params = []
    output_params = []

    for i in range(len(mp4files)):
        if pathonly:
            out[mp4files_scrub[i]] = os.path.join(viddir, mp4files[i])
        else:
            logger.warning(
                "NOTE: Ignoring {} files numbered above {}".format(extensions, maxopt)
            )
            out[mp4files_scrub[i]] = imageio.get_reader(
                os.path.join(viddir, mp4files[i]),
                pixelformat=pixelformat,
                input_params=input_params,
                output_params=output_params,
            )

    return out


"""
LOAD EXP INFO
"""


def load_expdict(params: Dict, e: int, expdict: Dict, viddir=_DEFAULT_VIDDIR):
    """
    Load in camnames and video directories and label3d files for a single experiment
        during training.
    """
    _DEFAULT_NPY_DIR = "npy_volumes"
    exp = params.copy()
    exp = make_paths_safe(exp)
    exp["label3d_file"] = expdict["label3d_file"]
    exp["base_exp_folder"] = os.path.dirname(exp["label3d_file"])

    if "viddir" not in expdict:
        # if the videos are not at the _DEFAULT_VIDDIR, then it must
        # be specified in the io.yaml experiment portion
        exp["viddir"] = os.path.join(exp["base_exp_folder"], viddir)
    else:
        exp["viddir"] = expdict["viddir"]

    l3d_camnames = io.load_camnames(expdict["label3d_file"])
    if "camnames" in expdict:
        exp["camnames"] = expdict["camnames"]
    elif l3d_camnames is not None:
        exp["camnames"] = l3d_camnames

    # Use the camnames to find the chunks for each video
    chunks = {}
    for name in exp["camnames"]:
        if exp["vid_dir_flag"]:
            camdir = os.path.join(exp["viddir"], name)
        else:
            camdir = os.path.join(exp["viddir"], name)
            intermediate_folder = os.listdir(camdir)
            camdir = os.path.join(camdir, intermediate_folder[0])
        video_files = os.listdir(camdir)
        video_files = [
            f for f in video_files if (".mp4" in f) and (f[0] != "_") and f[0] != "."
        ]
        video_files = sorted(video_files, key=lambda x: int(x.split(".")[0]))
        chunks[str(e) + "_" + name] = np.sort(
            [int(x.split(".")[0]) for x in video_files]
        )
    exp["chunks"] = chunks

    # For npy volume training
    if params["use_npy"]:
        exp["npy_vol_dir"] = os.path.join(exp["base_exp_folder"], _DEFAULT_NPY_DIR)
    return exp


def load_all_exps(params: Dict):
    """
    For a given set of experiments, load in the experiment-related information
    """
    samples = []  # training sample identifiers
    datadict, datadict_3d, com3d_dict = {}, {}, {}  # labels
    cameras, camnames = {}, {}  # camera
    total_chunks = {}  # video chunks
    temporal_chunks = {}  # for temporal training

    pbar = tqdm(params["exp"], desc="EXP PROCESSING")

    for e, expdict in enumerate(pbar):

        # load basic exp info
        exp = load_expdict(
            params, e, expdict, _DEFAULT_VIDDIR,
        )

        # load corresponding 2D & 3D labels, COMs
        (
            exp,
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            com3d_dict_,
            temporal_chunks_,
        ) = do_COM_load(exp, expdict, e, params)

        (
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
            temporal_chunks,
        ) = serve_data_DANNCE.add_experiment(
            e,
            samples,
            datadict,
            datadict_3d,
            com3d_dict,
            samples_,
            datadict_,
            datadict_3d_,
            com3d_dict_,
            temporal_chunks,
            temporal_chunks_,
        )

        cameras[e] = cameras_
        camnames[e] = exp["camnames"]

        params["experiment"][e] = exp
        for name, chunk in exp["chunks"].items():
            total_chunks[name] = chunk

    samples = np.array(samples)

    return (
        samples,
        datadict,
        datadict_3d,
        com3d_dict,
        cameras,
        camnames,
        total_chunks,
        temporal_chunks,
    )


def load_all_com_exps(params: Dict, exps: List):
    """
    Load all COM experiments.
    """
    params["experiment"] = {}
    total_chunks = {}
    cameras = {}
    camnames = {}
    datadict = {}
    datadict_3d = {}
    samples = []
    for e, expdict in enumerate(exps):

        exp = load_expdict(params, e, expdict, _DEFAULT_VIDDIR)

        params["experiment"][e] = exp
        (
            samples_,
            datadict_,
            datadict_3d_,
            cameras_,
            _,
        ) = serve_data_DANNCE.prepare_data(
            params["experiment"][e], com_flag=not params["multi_mode"],
        )

        # No need to prepare any COM file (they don't exist yet).
        # We call this because we want to support multiple experiments,
        # which requires appending the experiment ID to each data object and key
        samples, datadict, datadict_3d, _, _ = serve_data_DANNCE.add_experiment(
            e,
            samples,
            datadict,
            datadict_3d,
            {},
            samples_,
            datadict_,
            datadict_3d_,
            {},
        )

        cameras[e] = cameras_
        camnames[e] = params["experiment"][e]["camnames"]
        for name, chunk in exp["chunks"].items():
            total_chunks[name] = chunk

    samples = np.array(samples)

    return samples, datadict, datadict_3d, cameras, camnames, total_chunks


def do_COM_load(exp: Dict, expdict: Dict, e, params: Dict, training=True):
    """Load and process COMs.

    Args:
        exp (Dict): Parameters dictionary for experiment
        expdict (Dict): Experiment specific overrides (e.g. com_file, vid_dir)
        e (TYPE): Description
        params (Dict): Parameters dictionary.
        training (bool, optional): If true, load COM for training frames.

    Returns:
        TYPE: Description
        exp, samples_, datadict_, datadict_3d_, cameras_, com3d_dict_

    Raises:
        Exception: Exception when invalid com file format.
    """
    (
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
        temporal_chunks,
    ) = serve_data_DANNCE.prepare_data(
        exp,
        prediction=not training,
        predict_labeled_only=params["predict_labeled_only"],
        valid=(e in params["valid_exp"]) if params["valid_exp"] is not None else False,
        support=(e in params["support_exp"])
        if params["support_exp"] is not None
        else False,
        downsample=params["downsample"],
        return_full2d=params["return_full2d"]
        if "return_full2d" in params.keys()
        else False,
    )

    # If there is "clean" data (full marker set), can take the
    # 3D COM from the labels
    if exp["com_fromlabels"] and training:
        logger.info("For experiment {}, calculating 3D COM from labels".format(e))
        com3d_dict_ = deepcopy(datadict_3d_)
        for key in com3d_dict_.keys():
            com3d_dict_[key] = np.nanmean(datadict_3d_[key], axis=1, keepdims=True)
    elif "com_file" in expdict and expdict["com_file"] is not None:
        exp["com_file"] = expdict["com_file"]
        if ".mat" in exp["com_file"]:
            c3dfile = sio.loadmat(exp["com_file"])
            com3d_dict_ = check_COM_load(c3dfile, "com", params["medfilt_window"])

            # unlabeled frames sampling
            # currently assume that all training com files are named `instance%ncom3d.mat`
            # and there are exactly two instances
            if (
                training
                and (params["unlabeled_sampling"] is not None)
                and ("instance" in exp["com_file"].split("/")[-1])
            ):
                unlabeled_sampling = params["unlabeled_sampling"]
                sampling_num = unlabeled_sampling
                if not isinstance(unlabeled_sampling, int):
                    assert unlabeled_sampling in ["equal"]
                    sampling_num = len(samples_)

                # read com
                if "instance0" in exp["com_file"]:
                    pair_com_file = exp["com_file"].replace("instance0", "instance1")
                else:
                    pair_com_file = exp["com_file"].replace("instance1", "instance0")

                selected_samples = []
                if params.get("valid_exp") is not None and e in params["valid_exp"]:
                    selected_samples = []
                elif os.path.exists(pair_com_file):
                    comfile = sio.loadmat(pair_com_file)
                    c3d = comfile["com"]
                    sampleIDs = np.squeeze(comfile["sampleID"])
                    com_dist = np.sum((c3d - c3dfile["com"]) ** 2, axis=1)
                    com_dist = np.squeeze(np.sqrt(com_dist))  # [N]
                    # only sample from close interaction? hard coded distance threshold for now
                    indices_below_thres = np.where(com_dist < 120)[0]
                    indices_existing = [
                        i for i in range(len(sampleIDs)) if sampleIDs[i] in samples_
                    ]
                    indices_below_thres = list(
                        set(indices_below_thres) - set(indices_existing)
                    )
                    sampling_num = min(sampling_num, len(indices_below_thres))
                    selected_indices = np.random.choice(
                        indices_below_thres, size=sampling_num, replace=False
                    )
                    selected_samples = sampleIDs[selected_indices]

                logger.info(
                    "Unlabeled sampling: EXP {} added {} samples".format(
                        e, len(selected_samples)
                    )
                )
                samples_ = list(samples_) + list(selected_samples)
                samples_ = sorted(samples_)
                samples_ = np.array(samples_)

                nKeypoints = params["n_channels_out"]
                for i in range(len(selected_samples)):
                    samp = selected_samples[i]
                    data, frames = {}, {}
                    for j in range(len(params["camnames"])):
                        frames[params["camnames"][j]] = samp
                        data[params["camnames"][j]] = np.nan * np.ones((2, nKeypoints))
                    datadict_[samp] = {"data": data, "frames": frames}
                    datadict_3d_[samp] = np.nan * np.ones((3, nKeypoints))

        elif ".pickle" in exp["com_file"]:
            datadict_, com3d_dict_ = serve_data_DANNCE.prepare_COM(
                exp["com_file"],
                datadict_,
                comthresh=params["comthresh"],
                weighted=params["weighted"],
                camera_mats=cameras_,
                method=params["com_method"],
            )
            if params["medfilt_window"] is not None:
                raise Exception(
                    "Sorry, median filtering a com pickle is not yet supported. Please use a com3d.mat or *dannce.mat file instead"
                )
        else:
            raise Exception("Not a valid com file format")
    else:
        # Then load COM from the label3d file
        exp["com_file"] = expdict["label3d_file"]
        c3dfile = io.load_com(exp["com_file"])
        com3d_dict_ = check_COM_load(c3dfile, "com3d", params["medfilt_window"])

    # print("Experiment {} using com3d: {}".format(e, exp["com_file"]))

    # if params["medfilt_window"] is not None:
    #     print(
    #         "Median filtering COM trace with window size {}".format(
    #             params["medfilt_window"]
    #         )
    #     )

    # Remove any 3D COMs that are beyond the confines off the 3D arena
    do_cthresh = True if exp["cthresh"] is not None else False

    pre = len(samples_)
    samples_ = serve_data_DANNCE.remove_samples_com(
        samples_, com3d_dict_, rmc=do_cthresh, cthresh=exp["cthresh"],
    )
    # msg = "Removed {} samples from the dataset because they either had COM positions over cthresh, or did not have matching sampleIDs in the COM file"
    # print(msg.format(pre - len(samples_)))

    return (
        exp,
        samples_,
        datadict_,
        datadict_3d_,
        cameras_,
        com3d_dict_,
        temporal_chunks,
    )


def check_COM_load(c3dfile: Dict, kkey: Text, win_size: int):
    """Check that the COM file is of the appropriate format, and filter it.

    Args:
        c3dfile (Dict): Loaded com3d dictionary.
        kkey (Text): Key to use for extracting com.
        wsize (int): Window size.

    Returns:
        Dict: Dictionary containing com data.
    """
    c3d = c3dfile[kkey]

    # do a median filter on the COM traces if indicated
    if win_size is not None:
        if win_size % 2 == 0:
            win_size += 1
            # print("medfilt_window was not odd, changing to: {}".format(win_size))

        from scipy.signal import medfilt

        if len(c3d.shape) == 3:
            c3d = medfilt(c3d, (win_size, 1, 1))
        else:
            c3d = medfilt(c3d, (win_size, 1))

    c3dsi = np.squeeze(c3dfile["sampleID"])
    com3d_dict = {s: c3d[i] for (i, s) in enumerate(c3dsi)}
    return com3d_dict


def trim_COM_pickle(fpath, start_sample, end_sample, opath=None):
    """Trim dictionary entries to the range [start_sample, end_sample].

    spath is the output path for saving the trimmed COM dictionary, if desired
    """
    with open(fpath, "rb") as f:
        save_data = cPickle.load(f)
    sd = {}

    for key in save_data:
        if key >= start_sample and key <= end_sample:
            sd[key] = save_data[key]

    with open(opath, "wb") as f:
        cPickle.dump(sd, f)
    return sd


"""
DATA SPLITS
"""


def make_data_splits(
    samples, params, results_dir, num_experiments, temporal_chunks=None
):
    """
    Make train/validation splits from list of samples, or load in a specific
        list of sampleIDs if desired.
    """
    # TODO: Switch to .mat from .pickle so that these lists are easier to read
    # and change.

    partition = {}
    if params.get("use_temporal", False):
        if params["load_valid"] is None:
            assert (
                temporal_chunks != None
            ), "If use temporal, do partitioning over chunks."
            v = params["num_validation_per_exp"]
            # fix random seeds
            if params["data_split_seed"] is not None:
                np.random.seed(params["data_split_seed"])

            valid_chunks, train_chunks = [], []
            if params["valid_exp"] is not None and v > 0:
                for e in range(num_experiments):
                    if e in params["valid_exp"]:
                        v = params["num_validation_per_exp"]
                        if v > len(temporal_chunks[e]):
                            v = len(temporal_chunks[e])
                            logger.warning(
                                "Setting all {} samples in experiment {} for validation purpose.".format(
                                    v, e
                                )
                            )

                        valid_chunk_idx = sorted(
                            np.random.choice(len(temporal_chunks[e]), v, replace=False)
                        )
                        valid_chunks += list(
                            np.array(temporal_chunks[e])[valid_chunk_idx]
                        )
                        train_chunks += list(
                            np.delete(temporal_chunks[e], valid_chunk_idx, 0)
                        )
                    else:
                        train_chunks += temporal_chunks[e]
            elif v > 0:
                for e in range(num_experiments):
                    valid_chunk_idx = sorted(
                        np.random.choice(len(temporal_chunks[e]), v, replace=False)
                    )
                    valid_chunks += list(np.array(temporal_chunks[e])[valid_chunk_idx])
                    train_chunks += list(
                        np.delete(temporal_chunks[e], valid_chunk_idx, 0)
                    )
            elif params["valid_exp"] is not None:
                raise Exception("Need to set num_validation_per_exp in using valid_exp")
            else:
                for e in range(num_experiments):
                    train_chunks += list(temporal_chunks[e])

            train_expts = np.arange(num_experiments)
            logger.info("TRAIN EXPTS: {}".format(train_expts))

            if isinstance(params["training_fraction"], float):
                assert (params["training_fraction"] < 1.0) & (
                    params["training_fraction"] > 0
                )

                # load in the training samples
                labeled_train_samples = np.load(
                    "train_samples/baseline.pickle", allow_pickle=True
                )
                # labeled_train_chunks = [labeled_train_samples[i:i+params["temporal_chunk_size"]] for i in range(0, len(labeled_train_samples), params["temporal_chunk_size"])]
                n_chunks = len(labeled_train_samples)
                # do the selection from
                labeled_train_idx = sorted(
                    np.random.choice(
                        n_chunks,
                        int(n_chunks * params["training_fraction"]),
                        replace=False,
                    )
                )
                idxes_to_be_removed = list(
                    set(range(n_chunks)) - set(labeled_train_idx)
                )
                train_samples_to_be_removed = [
                    labeled_train_samples[i] for i in idxes_to_be_removed
                ]

                new_train_chunks = []
                for chunk in train_chunks:
                    if chunk[2] not in train_samples_to_be_removed:
                        new_train_chunks.append(chunk)
                train_chunks = new_train_chunks

            train_sampleIDs = list(np.concatenate(train_chunks))
            try:
                valid_sampleIDs = list(np.concatenate(valid_chunks))
            except:
                valid_sampleIDs = []

            partition["train_sampleIDs"], partition["valid_sampleIDs"] = (
                train_sampleIDs,
                valid_sampleIDs,
            )

        else:
            # Load validation samples from elsewhere
            with open(
                os.path.join(params["load_valid"], "val_samples.pickle"), "rb"
            ) as f:
                partition["valid_sampleIDs"] = cPickle.load(f)
            partition["train_sampleIDs"] = [
                f for f in samples if f not in partition["valid_sampleIDs"]
            ]

        chunk_size = len(temporal_chunks[0][0])
        partition["train_chunks"] = [
            np.arange(i, i + chunk_size)
            for i in range(0, len(partition["train_sampleIDs"]), chunk_size)
        ]
        partition["valid_chunks"] = [
            np.arange(i, i + chunk_size)
            for i in range(0, len(partition["valid_sampleIDs"]), chunk_size)
        ]
        # breakpoint()
        # Save train/val inds
        with open(os.path.join(results_dir, "val_samples.pickle"), "wb") as f:
            cPickle.dump(partition["valid_sampleIDs"], f)

        with open(os.path.join(results_dir, "train_samples.pickle"), "wb") as f:
            cPickle.dump(partition["train_sampleIDs"], f)
        return partition

    if params["load_valid"] is None:
        # Set random seed if included in params
        if params["data_split_seed"] is not None:
            np.random.seed(params["data_split_seed"])

        all_inds = np.arange(len(samples))

        # extract random inds from each set for validation
        v = params["num_validation_per_exp"]
        valid_inds = []
        if params["valid_exp"] is not None and v > 0:
            all_valid_inds = []
            for e in params["valid_exp"]:
                tinds = [
                    i for i in range(len(samples)) if int(samples[i].split("_")[0]) == e
                ]
                all_valid_inds = all_valid_inds + tinds

                # enable full validation experiments
                # by specifying params["num_validation_per_exp"] > number of samples
                v = params["num_validation_per_exp"]
                if v > len(tinds):
                    v = len(tinds)
                    logger.info(
                        "Setting all {} samples in experiment {} for validation purpose.".format(
                            v, e
                        )
                    )

                valid_inds = valid_inds + list(
                    np.random.choice(tinds, (v,), replace=False)
                )
                valid_inds = list(np.sort(valid_inds))

            train_inds = list(
                set(all_inds) - set(all_valid_inds)
            )  # [i for i in all_inds if i not in all_valid_inds]
            if isinstance(params["training_fraction"], float):
                assert (params["training_fraction"] < 1.0) & (
                    params["training_fraction"] > 0
                )
                n_samples = len(train_inds)
                train_inds_idx = sorted(
                    np.random.choice(
                        n_samples,
                        int(n_samples * params["training_fraction"]),
                        replace=False,
                    )
                )
                train_inds = [train_inds[i] for i in train_inds_idx]

        elif v > 0:  # if 0, do not perform validation
            for e in range(num_experiments):
                tinds = [
                    i for i in range(len(samples)) if int(samples[i].split("_")[0]) == e
                ]
                valid_inds = valid_inds + list(
                    np.random.choice(tinds, (v,), replace=False)
                )
                valid_inds = list(np.sort(valid_inds))

            train_inds = [i for i in all_inds if i not in valid_inds]
        elif params["valid_exp"] is not None:
            raise Exception("Need to set num_validation_per_exp in using valid_exp")
        else:
            train_inds = all_inds

        assert (set(valid_inds) & set(train_inds)) == set()
        train_samples = samples[train_inds]
        train_inds = []
        if params["valid_exp"] is not None:
            train_expts = [
                f for f in range(num_experiments) if f not in params["valid_exp"]
            ]
        else:
            train_expts = np.arange(num_experiments)

        logger.info("TRAIN EXPTS: {}".format(train_expts))

        if params["num_train_per_exp"] is not None:
            # Then sample randomly without replacement from training sampleIDs
            for e in train_expts:
                tinds = [
                    i
                    for i in range(len(train_samples))
                    if int(train_samples[i].split("_")[0]) == e
                ]
                # print(e)
                # print(len(tinds))
                train_inds = train_inds + list(
                    np.random.choice(
                        tinds, (params["num_train_per_exp"],), replace=False
                    )
                )
                train_inds = list(np.sort(train_inds))
        else:
            train_inds = np.arange(len(train_samples))

        partition["valid_sampleIDs"] = samples[valid_inds]
        partition["train_sampleIDs"] = train_samples[train_inds]
    else:
        # Load validation samples from elsewhere
        with open(os.path.join(params["load_valid"], "val_samples.pickle"), "rb",) as f:
            partition["valid_sampleIDs"] = cPickle.load(f)
        partition["train_sampleIDs"] = [
            f for f in samples if f not in partition["valid_sampleIDs"]
        ]
    # Save train/val inds
    with open(os.path.join(results_dir, "val_samples.pickle"), "wb") as f:
        cPickle.dump(partition["valid_sampleIDs"], f)

    with open(os.path.join(results_dir, "train_samples.pickle"), "wb") as f:
        cPickle.dump(partition["train_sampleIDs"], f)

    # Reset any seeding so that future batch shuffling, etc. are not tied to this seed
    if params["data_split_seed"] is not None:
        np.random.seed()

    return partition


def resplit_social(partition: Dict):
    # the partition needs to be aligned for both animals
    # for now, manually put exps as consecutive pairs,
    # i.e. [exp1_instance0, exp1_instance1, exp2_instance0, exp2_instance1, ...]
    new_partition = {"train_sampleIDs": [], "valid_sampleIDs": []}
    pairs = {"train_pairs": [], "valid_pairs": []}

    all_sampleIDs = np.concatenate(
        (partition["train_sampleIDs"], partition["valid_sampleIDs"])
    )
    for samp in partition["train_sampleIDs"]:
        exp_id = int(samp.split("_")[0])
        if exp_id % 2 == 0:
            paired = samp.replace(f"{exp_id}_", f"{exp_id+1}_")
            new_partition["train_sampleIDs"].append(samp)
            new_partition["train_sampleIDs"].append(paired)
            pairs["train_pairs"].append([samp, paired])

    new_partition["train_sampleIDs"] = np.array(
        sorted(new_partition["train_sampleIDs"])
    )
    new_partition["valid_sampleIDs"] = np.array(
        sorted(list(set(all_sampleIDs) - set(new_partition["train_sampleIDs"])))
    )

    for samp in new_partition["valid_sampleIDs"]:
        exp_id = int(samp.split("_")[0])
        if exp_id % 2 == 0:
            paired = samp.replace(f"{exp_id}_", f"{exp_id+1}_")
            pairs["valid_pairs"].append([samp, paired])

    return new_partition, pairs


def align_social_data(X, X_grid, y, aux, n_animals=2):
    X = X.reshape((n_animals, -1, *X.shape[1:]))
    X_grid = X_grid.reshape((n_animals, -1, *X_grid.shape[1:]))
    y = y.reshape((n_animals, -1, *y.shape[1:]))
    if aux is not None:
        aux = aux.reshape((n_animals, -1, *aux.shape[1:]))

    X = np.transpose(X, (1, 0, 2, 3, 4, 5))
    X_grid = np.transpose(X_grid, (1, 0, 2, 3))
    y = np.transpose(y, (1, 0, 2, 3))
    if aux is not None:
        aux = np.transpose(aux, (1, 0, 2, 3, 4, 5))

    return X, X_grid, y, aux


def remove_samples_npy(npydir: Dict, samples: List, params: Dict):
    """
    Remove any samples from sample list if they do not have corresponding volumes in the image
        or grid directories
    """
    samps = []
    for e in npydir.keys():
        imvol = os.path.join(npydir[e], "image_volumes")
        gridvol = os.path.join(npydir[e], "grid_volumes")
        ims = os.listdir(imvol)
        grids = os.listdir(gridvol)
        npysamps = [
            "0_" + f.split("_")[1] + ".npy"
            for f in samples
            if int(f.split("_")[0]) == e
        ]

        goodsamps = list(set(npysamps) & set(ims) & set(grids))

        samps = samps + [
            str(e) + "_" + f.split("_")[1].split(".")[0] for f in goodsamps
        ]

        sampdiff = len(npysamps) - len(goodsamps)

        # import pdb; pdb.set_trace()
        logger.warning(
            "Removed {} samples from {} because corresponding image or grid files could not be found".format(
                sampdiff, params["experiment"][e]["label3d_file"]
            )
        )

    return np.array(samps)


def reselect_training(partition: Dict, datadict_3d: Dict, frac: Union[float, int]):
    """
    Resample the training set according to the specified fraction,
    or by a certain number of samples.
    """
    samples = partition["train_sampleIDs"]
    unlabeled_samples = []
    for samp in samples:
        if np.isnan(datadict_3d[samp]).all():
            unlabeled_samples.append(samp)

    labeled_samples = list(set(samples) - set(unlabeled_samples))
    n_unlabeled = len(unlabeled_samples)
    n_labeled = len(labeled_samples)

    # the fraction number can either be a float <= 1 or an explicit integer
    if isinstance(frac, float):
        n_selected = np.minimum(
            int(frac * n_labeled), n_unlabeled
        )  # int(n_unlabeled*frac)
    else:
        n_selected = int(frac)

    unlabeled_samples = list(
        np.random.choice(unlabeled_samples, n_selected, replace=False)
    )

    partition["train_sampleIDs"] = sorted(unlabeled_samples + labeled_samples)

    logger.info(
        "***LABELED: UNLABELED = {}:{}".format(
            len(labeled_samples), len(unlabeled_samples)
        )
    )

    return partition


"""
PRELOAD DATA INTO MEMORY
"""


def load_volumes_into_mem(
    params: Dict,
    partition: Dict,
    n_cams: int,
    generator: torch.utils.data.Dataset,
    train: bool = True,
    silhouette: bool = False,
    social: bool = False,
):
    """
    Generate and load the training/validation volume data into memory using the data generator.
    This is different from directly reading from precached npy files.
    """
    n_samples = (
        len(partition["train_sampleIDs"])
        if train
        else len(partition["valid_sampleIDs"])
    )
    message = (
        "Loading training data into memory"
        if train
        else "Loading validation data into memory"
    )
    gridsize = tuple([params["nvox"]] * 3)

    # initialize vars
    if silhouette:
        X = np.empty((n_samples, *gridsize, n_cams), dtype="float32")
    else:
        X = np.empty(
            (n_samples, *gridsize, params["chan_num"] * n_cams), dtype="float32"
        )
    logger.info(message)

    X_grid = np.empty((n_samples, params["nvox"] ** 3, 3), dtype="float32")
    y = None
    if params["expval"]:
        if not silhouette:
            y = np.empty((n_samples, 3, params["n_channels_out"]), dtype="float32")
    else:
        y = np.empty((n_samples, *gridsize, params["n_channels_out"]), dtype="float32")

    # load data from generator
    if social:
        X = np.reshape(X, (2, -1, *X.shape[1:]))
        if X_grid is not None:
            X_grid = np.reshape(X_grid, (2, -1, *X_grid.shape[1:]))
        if y is not None:
            y = np.reshape(y, (2, -1, *y.shape[1:]))

        for i in tqdm(range(n_samples // 2)):
            rr = generator.__getitem__(i)
            for j in range(2):
                vol = rr[0][0][j]
                if not silhouette:
                    X[j, i] = vol
                    X_grid[j, i], y[j, i] = rr[0][1][j], rr[1][0][j]
                else:
                    X[j, i] = vol[:, :, :, ::3]  # extract_3d_sil(vol)
                    X_grid[j, i] = rr[0][1][j]

        X = np.reshape(X, (-1, *X.shape[2:]))
        # if silhouette:
        #     save_volumes_into_tif(params, './sil3d', X, np.arange(n_samples), n_cams, logger)
        if X_grid is not None:
            X_grid = np.reshape(X_grid, (-1, *X_grid.shape[2:]))
        if y is not None:
            y = np.reshape(y, (-1, *y.shape[2:]))

    else:
        for i in tqdm(range(n_samples)):
            rr = generator.__getitem__(i)
            if params["expval"]:
                vol = rr[0][0][0]
                if not silhouette:
                    X[i] = vol
                    X_grid[i], y[i] = rr[0][1], rr[1][0]
                else:
                    X[i] = vol[:, :, :, ::3]  # extract_3d_sil(vol)
                    X_grid[i] = rr[0][1]
            else:
                X[i], y[i] = rr[0][0], rr[1][0]

    if silhouette:
        logger.info("Now loading binary silhouettes")
        return None, X_grid, X

    return X, X_grid, y


def save_volumes_into_npy(
    params: Dict,
    npy_generator: torch.utils.data.Dataset,
    missing_npydir: Dict,
    samples: List,
    silhouette=False
):
    """
    Cache the training volumes into npy files for later use.
    """
    logger.info("Generating missing npy files ...")
    pbar = tqdm(npy_generator.list_IDs)
    for i, samp in enumerate(pbar):
        fname = "0_{}.npy".format(samp.split("_")[1])
        rr = npy_generator.__getitem__(i)
        # print(i, end="\r")

        if params["is_social_dataset"]:
            for j in range(npy_generator.n_instances):
                exp = int(samp.split("_")[0]) + j
                save_root = missing_npydir[exp]

                if not silhouette:
                    X = rr[0][0][j].astype("uint8")
                    X_grid, y = rr[0][1][j], rr[1][0][j]

                    for savedir, data in zip(
                        ["image_volumes", "grid_volumes", "targets"], [X, X_grid, y]
                    ):
                        outdir = os.path.join(save_root, savedir, fname)
                        if not os.path.exists(outdir):
                            np.save(outdir, data)

                    if params["downscale_occluded_view"]:
                        np.save(
                            os.path.join(save_root, "occlusion_scores", fname),
                            rr[0][2][j],
                        )
                else:
                    # sil = extract_3d_sil(rr[0][0][j].astype("uint8"))
                    sil = rr[0][0][j].astype("uint8")[:, :, :, ::3]
                    np.save(os.path.join(save_root, "visual_hulls", fname), sil)

        else:
            exp = int(samp.split("_")[0])
            save_root = missing_npydir[exp]

            X, X_grid, y = rr[0][0][0].astype("uint8"), rr[0][1][0], rr[1][0]

            if not silhouette:
                for savedir, data in zip(
                    ["image_volumes", "grid_volumes", "targets"], [X, X_grid, y]
                ):
                    outdir = os.path.join(save_root, savedir, fname)
                    if not os.path.exists(outdir):
                        np.save(outdir, data)
            else:
                # sil = extract_3d_sil(X)
                sil = X[:, :, :, ::3]
                np.save(os.path.join(save_root, "visual_hulls", fname), sil)

    # samples = remove_samples_npy(npydir, samples, params)
    logger.info("{} samples ready for npy training.".format(len(samples)))


"""
DEBUG, VIS
"""


def write_debug(
    params: Dict,
    ims_train: np.ndarray,
    ims_valid: np.ndarray,
    y_train: np.ndarray,
    # model,
    trainData: bool = True,
):
    """Factoring re-used debug output code.

    Args:
        params (Dict): Parameters dictionary
        ims_train (np.ndarray): Training images
        ims_valid (np.ndarray): Validation images
        y_train (np.ndarray): Training targets
        model (Model): Model
        trainData (bool, optional): If True use training data for debug. Defaults to True.
    """

    def plot_out(imo, lo, imn):
        plot_markers_2d(norm_im(imo), lo, newfig=False)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        imname = imn
        plt.savefig(os.path.join(debugdir, imname), bbox_inches="tight", pad_inches=0)

    if params["debug"] and not params["multi_mode"]:

        if trainData:
            outdir = "debug_im_out"
            ims_out = ims_train
            label_out = y_train
        # else:
        #     outdir = "debug_im_out_valid"
        #     ims_out = ims_valid
        #     label_out = model.predict(ims_valid, batch_size=1)

        # Plot all training images and save
        # create new directory for images if necessary
        debugdir = os.path.join(params["com_train_dir"], outdir)
        logger.success("Saving debug images to: " + debugdir)
        if not os.path.exists(debugdir):
            os.makedirs(debugdir)

        plt.figure()

        for i in range(ims_out.shape[0]):
            plt.cla()
            if params["mirror"]:
                for j in range(label_out.shape[-1]):
                    plt.cla()
                    plot_out(
                        ims_out[i],
                        label_out[i, :, :, j : j + 1],
                        str(i) + "_cam_" + str(j) + ".png",
                    )
            else:
                plot_out(ims_out[i], label_out[i], str(i) + ".png")

    elif params["debug"] and params["multi_mode"]:
        logger.error("Note: Cannot output debug information in COM multi-mode")


def save_volumes_into_tif(
    params: Dict,
    tifdir: str,
    X: np.ndarray,
    sampleIDs: List,
    n_cams: int,
):
    """
    Save a batch of 3D volumes into tif format so that they can be visualized.
    """
    if not os.path.exists(tifdir):
        os.makedirs(tifdir)
    logger.info("Dump training volumes to {}".format(tifdir))
    for i in range(X.shape[0]):
        for j in range(n_cams):
            im = X[
                i, :, :, :, j * params["chan_num"] : (j + 1) * params["chan_num"],
            ]
            im = norm_im(im) * 255
            im = im.astype("uint8")
            of = os.path.join(tifdir, str(sampleIDs[i]) + "_cam" + str(j) + ".tif",)
            imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))


def generate_visual_hull(
    aux: np.ndarray,
    sampleIDs: List,
    savedir: str = "./visual_hull",
):
    """
    Generate visual hulls from the auxiliary data using marching cubes algorithm. 
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for i in range(aux.shape[0]):
        intersection = np.squeeze(aux[i].astype(np.float32))

        # apply marching cubes algorithm
        verts, faces, normals, values = measure.marching_cubes(intersection, 0.0)
        # print('Number of vertices: ', verts.shape[0])

        # save predictions
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor("k")
        ax.add_collection3d(mesh)

        min_limit, max_limit = np.min(verts), np.max(verts)

        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)
        ax.set_zlim(min_limit, max_limit)

        of = os.path.join(savedir, sampleIDs[i])
        fig.savefig(of)
        plt.close(fig)


"""
SAVE TRAIN
"""


def save_params_pickle(params: Dict):
    """
    save copy of params as pickle for reproducibility.
    """
    handle = open(os.path.join(params["dannce_train_dir"], "params.pickle"), "wb")
    pickle.dump(params, handle)

    return True


def save_params_yaml(params: Dict):
    """Save copy of params as yaml.

    Args:
        params (Dict): experiment parameters
    """
    # exclude keys for intermediate computations
    _exclude = ["experiment", "chunks"]
    params_to_save = {k: v for k, v in params.items() if k not in _exclude}
    handle = open(os.path.join(params["dannce_train_dir"], "params.yaml"), "w")
    yaml.dump(params_to_save, handle, default_flow_style=False, sort_keys=False)


def prepare_save_metadata(params: Dict):
    """
    To save metadata, i.e. the prediction param values associated with COM or DANNCE
        output, we need to convert loss and metrics and net into names, and remove
        the 'experiment' field
    """
    # Need to convert None to string but still want to conserve the metadat structure
    # format, so we don't want to convert the whole dict to a string
    meta = params.copy()
    if "experiment" in meta:
        del meta["experiment"]
    if "loss" in meta:
        try:
            meta["loss"] = [loss.__name__ for loss in meta["loss"]]
        except:
            meta["loss"] = list(meta["loss"].keys())

    meta = make_none_safe(meta.copy())
    return meta


def save_COM_dannce_mat(params, com3d, sampleID):
    """
    Instead of saving 3D COM to com3d.mat, save it into the dannce.mat file, which
    streamlines subsequent dannce access.
    """
    com = {}
    com["com3d"] = com3d
    com["sampleID"] = sampleID
    com["metadata"] = prepare_save_metadata(params)

    # Open dannce.mat file, add com and re-save
    logger.success("Saving COM predictions to " + params["label3d_file"])
    rr = sio.loadmat(params["label3d_file"])
    # For safety, save old file to temp and delete it at the end
    sio.savemat(params["label3d_file"] + ".temp", rr)
    rr["com"] = com
    sio.savemat(params["label3d_file"], rr)

    os.remove(params["label3d_file"] + ".temp")


def save_COM_checkpoint(
    save_data, results_dir, datadict_, cameras, params, file_name="com3d"
):
    """
    Saves COM pickle and matfiles

    """
    # Save undistorted 2D COMs and their 3D triangulations
    f = open(os.path.join(results_dir, file_name + ".pickle"), "wb")
    cPickle.dump(save_data, f)
    f.close()

    # We need to remove the eID in front of all the keys in datadict
    # for prepare_COM to run properly
    datadict_save = {}
    for key in datadict_:
        datadict_save[int(float(key.split("_")[-1]))] = datadict_[key]

    if params["n_instances"] > 1:
        if params["n_channels_out"] > 1:
            linking_method = "multi_channel"
        else:
            linking_method = "euclidean"
        _, com3d_dict = serve_data_DANNCE.prepare_COM_multi_instance(
            os.path.join(results_dir, file_name + ".pickle"),
            datadict_save,
            comthresh=0,
            weighted=False,
            camera_mats=cameras,
            linking_method=linking_method,
        )
    else:
        prepare_func = serve_data_DANNCE.prepare_COM
        _, com3d_dict = serve_data_DANNCE.prepare_COM(
            os.path.join(results_dir, file_name + ".pickle"),
            datadict_save,
            comthresh=0,
            weighted=False,
            camera_mats=cameras,
            method="median",
        )

    cfilename = os.path.join(results_dir, file_name + ".mat")
    logger.success("Saving 3D COM to {}".format(cfilename))
    samples_keys = list(com3d_dict.keys())

    if params["n_instances"] > 1:
        c3d = np.zeros((len(samples_keys), 3, params["n_instances"]))
    else:
        c3d = np.zeros((len(samples_keys), 3))

    for i in range(len(samples_keys)):
        c3d[i] = com3d_dict[samples_keys[i]]

    metadata = prepare_save_metadata(params)
    sio.savemat(
        cfilename, {"sampleID": samples_keys, "com": c3d, "metadata": metadata,},
    )

    # If multiple instances, additionally save to different files for each instance
    # keep consistent with `multi_gpu.py` com_merge()
    if params["n_instances"] > 1:  # and file_name == "com3d":
        for n_instance in range(params["n_instances"]):
            fn = os.path.join(
                results_dir, "instance" + str(n_instance) + file_name + ".mat",
            )
            sio.savemat(
                fn,
                {
                    "com": c3d[..., n_instance].squeeze(),
                    "sampleID": samples_keys,
                    "metadata": metadata,
                },
            )


def write_com_file(params, samples_, com3d_dict_):
    cfilename = os.path.join(params["dannce_predict_dir"], "com3d_used.mat")
    logger.success("Saving 3D COM to {}".format(cfilename))
    c3d_shape = com3d_dict_[samples_[0]].shape
    c3d = np.zeros((len(samples_), *c3d_shape))
    for i in range(len(samples_)):
        c3d[i] = com3d_dict_[samples_[i]]
    sio.savemat(cfilename, {"sampleID": samples_, "com": c3d})


def savedata_expval(
    fname,
    params,
    write=True,
    data=None,
    num_instances=1,
    num_markers=20,
    tcoord=True,
    pmax=False,
):
    """Save the expected values."""
    if data is None:
        f = open(fname, "rb")
        data = cPickle.load(f)
        f.close()

    d_coords = np.zeros((len(list(data.keys())), num_instances, 3, num_markers))
    t_coords = np.zeros((len(list(data.keys())), num_instances, 3, num_markers))
    sID = np.zeros((len(list(data.keys())),))
    p_max = np.zeros((len(list(data.keys())), num_instances, num_markers))

    for (i, key) in enumerate(data.keys()):
        d_coords[i] = data[key]["pred_coord"]
        if tcoord:
            t_coords[i] = np.reshape(
                data[key]["true_coord_nogrid"], (num_instances, 3, num_markers)
            )
        if pmax:
            p_max[i] = data[key]["pred_max"]
        sID[i] = data[key]["sampleID"]

        sdict = {
            "pred": d_coords,
            "data": t_coords,
            "p_max": p_max,
            "sampleID": sID,
            # "metadata": #prepare_save_metadata(params),
        }
    if write and data is None:
        sio.savemat(
            fname.split(".pickle")[0] + ".mat", sdict,
        )
    elif write and data is not None:
        sio.savemat(fname, sdict)

    return d_coords, t_coords, p_max, sID


def savedata_tomat(
    fname,
    params,
    vmin,
    vmax,
    nvox,
    write=True,
    data=None,
    num_markers=20,
    tcoord=True,
    tcoord_scale=True,
    addCOM=None,
):
    """Save pickled data to a mat file.

    From a save_data structure saved to a *.pickle file, save a matfile
        with useful variables for easier manipulation in matlab.
    Also return pred_out_world and other variables for plotting within jupyter
    """
    if data is None:
        f = open(fname, "rb")
        data = cPickle.load(f)
        f.close()

    d_coords = np.zeros((list(data.keys())[-1] + 1, 3, num_markers))
    t_coords = np.zeros((list(data.keys())[-1] + 1, 3, num_markers))
    p_max = np.zeros((list(data.keys())[-1] + 1, num_markers))
    log_p_max = np.zeros((list(data.keys())[-1] + 1, num_markers))
    sID = np.zeros((list(data.keys())[-1] + 1,))
    for (i, key) in enumerate(data.keys()):
        d_coords[i] = data[key]["pred_coord"]
        if tcoord:
            t_coords[i] = np.reshape(data[key]["true_coord_nogrid"], (3, num_markers))
        p_max[i] = data[key]["pred_max"]
        log_p_max[i] = data[key]["logmax"]
        sID[i] = data[key]["sampleID"]

    vsize = (vmax - vmin) / nvox
    # First, need to move coordinates over to centers of voxels
    pred_out_world = vmin + d_coords * vsize + vsize / 2

    if tcoord and tcoord_scale:
        t_coords = vmin + t_coords * vsize + vsize / 2

    if addCOM is not None:
        # We use the passed comdict to add back in the com, this is useful
        # if one wnats to bootstrap on these values for COMnet or otherwise
        for i in range(len(sID)):
            pred_out_world[i] = pred_out_world[i] + addCOM[int(sID)][:, np.newaxis]

    sdict = {
        "pred": pred_out_world,
        "data": t_coords,
        "p_max": p_max,
        "sampleID": sID,
        "log_pmax": log_p_max,
        # "metadata": prepare_save_metadata(params),
    }
    if write and data is None:
        sio.savemat(
            fname.split(".pickle")[0] + ".mat", sdict,
        )
    elif write and data is not None:
        sio.savemat(
            fname, sdict,
        )
    return pred_out_world, t_coords, p_max, log_p_max, sID


"""
IMAGE OPS (should be moved to ops)
"""


def __initAvgMax(t, g, o, params):
    """
    Helper function for creating 3D targets
    """
    gridsize = tuple([params["nvox"]] * 3)
    g = np.reshape(g, (-1, *gridsize, 3),)

    for i in range(o.shape[0]):
        for j in range(o.shape[-1]):
            o[i, ..., j] = np.exp(
                -(
                    (g[i, ..., 1] - t[i, 1, j]) ** 2
                    + (g[i, ..., 0] - t[i, 0, j]) ** 2
                    + (g[i, ..., 2] - t[i, 2, j]) ** 2
                )
                / (2 * params["sigma"] ** 2)
            )

    return o


def batch_rgb2gray(imstack):
    """Convert to gray image-wise.

    batch dimension is first.
    """
    grayim = np.zeros((imstack.shape[0], imstack.shape[1], imstack.shape[2]), "float32")
    for i in range(grayim.shape[0]):
        grayim[i] = rgb2gray(imstack[i].astype("uint8"))
    return grayim


def return_tile(imstack, fac=2):
    """Crop a larger image into smaller tiles without any overlap."""
    height = imstack.shape[1] // fac
    width = imstack.shape[2] // fac
    out = np.zeros(
        (imstack.shape[0] * fac * fac, height, width, imstack.shape[3]), "float32"
    )
    cnt = 0
    for i in range(imstack.shape[0]):
        for j in np.arange(0, imstack.shape[1], height):
            for k in np.arange(0, imstack.shape[2], width):
                out[cnt, :, :, :] = imstack[i, j : j + height, k : k + width, :]
                cnt = cnt + 1
    return out


def tile2im(imstack, fac=2):
    """Reconstruct lagrer image from tiled data."""
    height = imstack.shape[1]
    width = imstack.shape[2]
    out = np.zeros(
        (imstack.shape[0] // (fac * fac), height * fac, width * fac, imstack.shape[3]),
        "float32",
    )
    cnt = 0
    for i in range(out.shape[0]):
        for j in np.arange(0, out.shape[1], height):
            for k in np.arange(0, out.shape[2], width):
                out[i, j : j + height, k : k + width, :] = imstack[cnt]
                cnt += 1
    return out


def downsample_batch(imstack, fac=2, method="PIL"):
    """Downsample each image in a batch."""

    if method == "PIL":
        out = np.zeros(
            (
                imstack.shape[0],
                int(imstack.shape[1] / fac),
                int(imstack.shape[2] / fac),
                imstack.shape[3],
            ),
            "float32",
        )
        if out.shape[-1] == 3:
            # this is just an RGB image, so no need to loop over channels with PIL
            for i in range(imstack.shape[0]):
                out[i] = np.array(
                    PIL.Image.fromarray(imstack[i].astype("uint8")).resize(
                        (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS
                    )
                )
        else:
            for i in range(imstack.shape[0]):
                for j in range(imstack.shape[3]):
                    out[i, :, :, j] = np.array(
                        PIL.Image.fromarray(imstack[i, :, :, j]).resize(
                            (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS
                        )
                    )

    elif method == "dsm":
        out = np.zeros(
            (
                imstack.shape[0],
                imstack.shape[1] // fac,
                imstack.shape[2] // fac,
                imstack.shape[3],
            ),
            "float32",
        )
        for i in range(imstack.shape[0]):
            for j in range(imstack.shape[3]):
                out[i, :, :, j] = dsm(imstack[i, :, :, j], (fac, fac))

    elif method == "nn":
        out = imstack[:, ::fac, ::fac]

    elif fac > 1:
        raise Exception("Downfac > 1. Not a valid downsampling method")

    return out


def batch_maximum(imstack):
    """Find the location of the maximum for each image in a batch."""
    maxpos = np.zeros((imstack.shape[0], 2))
    for i in range(imstack.shape[0]):
        if np.isnan(imstack[i, 0, 0]):
            maxpos[i, 0] = np.nan
            maxpos[i, 1] = np.nan
        else:
            ind = np.unravel_index(
                np.argmax(np.squeeze(imstack[i]), axis=None),
                np.squeeze(imstack[i]).shape,
            )
            maxpos[i, 0] = ind[1]
            maxpos[i, 1] = ind[0]
    return maxpos


def cropcom(im, com, size=512):
    """Crops single input image around the coordinates com."""
    minlim_r = int(np.round(com[1])) - size // 2
    maxlim_r = int(np.round(com[1])) + size // 2
    minlim_c = int(np.round(com[0])) - size // 2
    maxlim_c = int(np.round(com[0])) + size // 2

    diff = (minlim_r, maxlim_r, minlim_c, maxlim_c)
    crop_dim = (np.max([minlim_r, 0]), maxlim_r, np.max([minlim_c, 0]), maxlim_c)

    out = im[crop_dim[0] : crop_dim[1], crop_dim[2] : crop_dim[3], :]

    dim = out.shape[2]

    # pad with zeros if region ended up outside the bounds of the original image
    if minlim_r < 0:
        out = np.concatenate(
            (np.zeros((abs(minlim_r), out.shape[1], dim)), out), axis=0
        )
    if maxlim_r > im.shape[0]:
        out = np.concatenate(
            (out, np.zeros((maxlim_r - im.shape[0], out.shape[1], dim))), axis=0
        )
    if minlim_c < 0:
        out = np.concatenate(
            (np.zeros((out.shape[0], abs(minlim_c), dim)), out), axis=1
        )
    if maxlim_c > im.shape[1]:
        out = np.concatenate(
            (out, np.zeros((out.shape[0], maxlim_c - im.shape[1], dim))), axis=1
        )

    return out, diff


def plot_markers_2d(im, markers, newfig=True):
    """Plot markers in two dimensions."""

    if newfig:
        plt.figure()
    plt.imshow((im - np.min(im)) / (np.max(im) - np.min(im)))

    for mark in range(markers.shape[-1]):
        ind = np.unravel_index(
            np.argmax(markers[:, :, mark], axis=None), markers[:, :, mark].shape
        )
        plt.plot(ind[1], ind[0], ".r")


def preprocess_3d(im_stack):
    """Easy inception-v3 style image normalization across a set of images."""
    im_stack /= 127.5
    im_stack -= 1.0
    return im_stack


def norm_im(im):
    """Normalize image."""
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def plot_markers_3d_torch(stack, nonan=True):
    """Return the 3d coordinates for each of the peaks in probability maps."""
    import torch

    n_mark = stack.shape[-1]
    index = stack.flatten(0, 2).argmax(dim=0).to(torch.int32)
    inds = unravel_index(index, stack.shape[:-1])
    if ~torch.any(torch.isnan(stack[0, 0, 0, :])) and (nonan or not nonan):
        x = inds[1]
        y = inds[0]
        z = inds[2]
    elif not nonan:
        x = inds[1]
        y = inds[0]
        z = inds[2]
        for mark in range(0, n_mark):
            if torch.isnan(stack[:, :, :, mark]):
                x[mark] = torch.nan
                y[mark] = torch.nan
                z[mark] = torch.nan
    return x, y, z


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def grid_channelwise_max(grid_):
    """Return the max value in each channel over a 3D volume.

    input--
        grid_: shape (nvox, nvox, nvox, nchannels)

    output--
        shape (nchannels,)
    """
    return np.max(np.max(np.max(grid_, axis=0), axis=0), axis=0)


def moment_3d(im, mesh, thresh=0):
    """Get the normalized spatial moments of the 3d image stack.

    inputs--
        im: 3d volume confidence map, one for each channel (marker)
            i.e. shape (nvox,nvox,nvox,nchannels)
        mesh: spatial coordinates for every position on im
        thresh: threshold applied to im before calculating moments
    """
    x = []
    y = []
    z = []
    for mark in range(im.shape[3]):
        # get normalized probabilities
        im_norm = (im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)) / np.sum(
            im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)
        )
        x.append(np.sum(mesh[0] * im_norm))
        y.append(np.sum(mesh[1] * im_norm))
        z.append(np.sum(mesh[2] * im_norm))
    return x, y, z


def get_peak_inds(map_):
    """Return the indices of the peak value of an n-d map."""
    return np.unravel_index(np.argmax(map_, axis=None), map_.shape)


def get_peak_inds_multi_instance(im, n_instances, window_size=10):
    """Return top n_instances local peaks through non-max suppression."""
    bw = im == maximum_filter(im, footprint=np.ones((window_size, window_size)))
    inds = np.argwhere(bw)
    vals = im[inds[:, 0], inds[:, 1]]
    idx = np.argsort(vals)[::-1]
    return inds[idx[:n_instances], :]


def get_marker_peaks_2d(stack):
    """Return the concatenated coordinates of all peaks for each map/marker."""
    x = []
    y = []
    for i in range(stack.shape[-1]):
        inds = get_peak_inds(stack[:, :, i])
        x.append(inds[1])
        y.append(inds[0])
    return x, y


def spatial_expval(map_):
    """Calculate the spatial expected value of the input.

    Note there is probably underflow here that I am ignoring, because this
    doesn't need to be *that* accurate
    """
    map_ = map_ / np.sum(map_)
    x, y = np.meshgrid(np.arange(map_.shape[1]), np.arange(map_.shape[0]))

    return np.sum(map_ * x), np.sum(map_ * y)


def spatial_var(map_):
    """Calculate the spatial variance of the input."""
    expx, expy = spatial_expval(map_)
    map_ = map_ / np.sum(map_)
    x, y = np.meshgrid(np.arange(map_.shape[1]), np.arange(map_.shape[0]))

    return np.sum(map_ * ((x - expx) ** 2 + (y - expy) ** 2))


def spatial_entropy(map_):
    """Calculate the spatial entropy of the input."""
    map_ = map_ / np.sum(map_)
    return -1 * np.sum(map_ * np.log(map_))


"""
SEGMENTATION
"""


def mask_to_bbox(mask):
    bounding_boxes = np.zeros((4,))
    y, x, _ = np.where(mask != 0)
    try:
        bounding_boxes[0] = np.min(x)
        bounding_boxes[1] = np.min(y)
        bounding_boxes[2] = np.max(x)
        bounding_boxes[3] = np.max(y)
    except:
        return bounding_boxes
    return bounding_boxes


def mask_iou(mask1, mask2):
    """ compute iou between two binary masks
    """
    intersection = np.sum(mask1 * mask2)
    if intersection == 0:
        return 0.0
    union = np.sum(np.logical_or(mask1, mask2).astype(np.uint8))
    return intersection / union


def mask_intersection(mask1, mask2):
    return mask1 * mask2


def bbox_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[0] - bb1[2]) * (bb1[1] - bb1[3])
    bb2_area = (bb2[0] - bb2[2]) * (bb2[1] - bb2[3])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def compute_support(coms, mask, support_region_size=10):
    counts = []
    for i in range(len(coms)):
        index = coms[i]  # .clone().cpu().int().numpy()
        sp_l = np.maximum(0, index[1] - support_region_size)
        sp_r = np.minimum(mask.shape[0], index[1] + support_region_size)
        sp_t = np.maximum(0, index[0] - support_region_size)
        sp_b = np.minimum(mask.shape[1], index[0] + support_region_size)

        count = np.sum(mask[int(sp_l) : int(sp_r), int(sp_t) : int(sp_b), 0])
        counts.append(count)
    return np.array(counts)


def extract_3d_sil(vol):
    """
    vol: [n_samples, H, W, D, C*n_cam]
    """
    vol[vol > 0] = 1
    vol = np.sum(vol, axis=-1, keepdims=True)

    # TODO: want max over each sample, instead of all
    upper_thres = np.max(vol)

    vol[vol < upper_thres] = 0
    vol[vol > 0] = 1

    print(
        "{}\% of silhouette training voxels are occupied".format(
            100 * np.sum(vol) / len(vol.ravel())
        )
    )
    return vol


def extract_3d_sil_soft(vol, keeprange=3):
    vol[vol > 0] = 1
    vol = np.sum(vol, axis=-1, keepdims=True)

    upper_thres = np.max(vol)
    lower_thres = upper_thres - keeprange
    vol[vol <= lower_thres] = 0
    vol[vol > 0] = (vol[vol > 0] - lower_thres) / keeprange

    print(
        "{}\% of silhouette training voxels are occupied".format(
            100 * np.sum((vol > 0)) / len(vol.ravel())
        )
    )
    return vol


def compute_bbox_from_3dmask(mask3d, grids):
    """
    mask3d: [N, H, W, D, 1]
    grid: [N, H*W*D, 3]
    """
    new_com3ds, new_dims = [], []
    for mask, grid in zip(mask3d, grids):
        mask = np.squeeze(mask)  # [H, W, D]
        h, w, d = np.where(mask)

        h_l, h_u = h.min(), h.max()
        w_l, w_u = w.min(), w.max()
        d_l, d_u = d.min(), d.max()

        corner1 = np.array([h_l, w_l, d_l])
        corner2 = np.array([h_u, w_u, d_u])
        mid_point = ((corner1 + corner2) / 2).astype(int)

        grid = np.reshape(grid, (*mask.shape, 3))

        new_com3d = grid[mid_point[0], mid_point[1], mid_point[2]]

        new_dim = (
            grid[corner2[0], corner2[1], corner2[2]]
            - grid[corner1[0], corner1[1], corner1[2]]
        )

        new_com3ds.append(new_com3d)
        new_dims.append(new_dim)

    new_com3ds = np.stack(new_com3ds, axis=0)
    new_dims = np.stack(new_dims, axis=0)

    return new_com3ds, new_dims


def create_new_labels(partition, old_com3ds, new_com3ds, new_dims, params):
    com3d_dict, dim_dict = {}, {}
    all_sampleIDs = [*partition["train_sampleIDs"], *partition["valid_sampleIDs"]]

    default_dim = np.array([(params["vmax"] - params["vmin"]) * 0.8] * 3)
    for sampleID, new_com, new_dim in zip(all_sampleIDs, new_com3ds, new_dims):
        if ((new_dim / 2) < params["vmax"] * 0.6).sum() > 0:
            com3d_dict[sampleID] = old_com3ds[sampleID]
            dim_dict[sampleID] = default_dim
        else:
            com3d_dict[sampleID] = new_com
            new_dim = 10 * (new_dim // 10) + 40
            dim_dict[sampleID] = new_dim
    return com3d_dict, dim_dict


def filter_com3ds(pairs, com3d_dict, datadict_3d, threshold=120):
    train_sampleIDs, valid_sampleIDs = [], []
    new_com3d_dict, new_datadict_3d = {}, {}

    for (a, b) in pairs["train_pairs"]:
        com1 = com3d_dict[a]
        com2 = com3d_dict[b]
        dist = np.sqrt(np.sum((com1 - com2) ** 2))
        if dist <= threshold:
            train_sampleIDs.append(a)
            new_com3d_dict[a] = (com1 + com2) / 2
            new_datadict_3d[a] = np.concatenate(
                (datadict_3d[a], datadict_3d[b]), axis=-1
            )

    for (a, b) in pairs["valid_pairs"]:
        com1 = com3d_dict[a]
        com2 = com3d_dict[b]
        dist = np.sqrt(np.sum((com1 - com2) ** 2))

        if dist <= threshold:
            valid_sampleIDs.append(a)
            new_com3d_dict[a] = (com1 + com2) / 2
            new_datadict_3d[a] = np.concatenate(
                (datadict_3d[a], datadict_3d[b]), axis=-1
            )

    partition = {}
    partition["train_sampleIDs"] = train_sampleIDs
    partition["valid_sampleIDs"] = valid_sampleIDs

    new_samples = np.array(sorted(train_sampleIDs + valid_sampleIDs))

    return partition, new_com3d_dict, new_datadict_3d, new_samples


def mask_coords_outside_volume(vmin, vmax, pose3d, anchor, n_chan):
    # compute relative distance to COM
    anchor_dist = pose3d - anchor
    x_in_vol = (anchor_dist[0] >= vmin) & (anchor_dist[0] <= vmax)
    y_in_vol = (anchor_dist[1] >= vmin) & (anchor_dist[1] <= vmax)
    z_in_vol = (anchor_dist[2] >= vmin) & (anchor_dist[2] <= vmax)

    in_vol = x_in_vol & y_in_vol & z_in_vol
    in_vol = np.stack([in_vol] * 3, axis=0)

    # if the other animal's partially in the volume, use masked nan
    # otherwise repeat the first animal
    nan_pose = np.empty_like(pose3d)
    nan_pose[:] = np.nan

    new_pose3d = np.where(in_vol, pose3d, nan_pose)

    if np.isnan(new_pose3d[:, n_chan:]).sum() == n_chan * 3:
        print("The other animal not in volume, repeat the primary.")
        new_pose3d[:, n_chan:] = new_pose3d[:, :n_chan]

    return new_pose3d


def prepare_joint_volumes(params, pairs, com3d_dict, datadict_3d):
    vmin, vmax = params["vmin"], params["vmax"]
    for k, v in pairs.items():
        for (vol1, vol2) in v:
            anchor1, anchor2 = com3d_dict[vol1], com3d_dict[vol2]
            anchor1, anchor2 = anchor1[:, np.newaxis], anchor2[:, np.newaxis]  # [3, 1]
            pose3d1, pose3d2 = datadict_3d[vol1], datadict_3d[vol2]

            n_chan = pose3d1.shape[-1]

            new_pose3d1 = np.concatenate((pose3d1, pose3d2), axis=-1)  # [3, 46]
            new_pose3d2 = np.concatenate((pose3d2, pose3d1), axis=-1)  # [3, 46]

            new_pose3d1 = mask_coords_outside_volume(
                vmin, vmax, new_pose3d1, anchor1, n_chan
            )
            new_pose3d2 = mask_coords_outside_volume(
                vmin, vmax, new_pose3d2, anchor2, n_chan
            )

            datadict_3d[vol1] = new_pose3d1
            datadict_3d[vol2] = new_pose3d2

    return datadict_3d


def _preprocess_numpy_input(x, data_format="channels_last", mode="torch"):
    """Preprocesses a Numpy array encoding a batch of images.
    Args:
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.
    Returns:
        Preprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype("float32", copy=False)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            mean = [103.939, 116.779, 123.68]
            std = None

    # Zero-center by mean pixel
    if data_format == "channels_first":
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
            else:
                x[:, 0, :, :] -= mean[0]
                x[:, 1, :, :] -= mean[1]
                x[:, 2, :, :] -= mean[2]
        if std is not None:
            x[:, 0, :, :] /= std[0]
            x[:, 1, :, :] /= std[1]
            x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x
