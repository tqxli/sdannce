"""Processing functions for dannce."""
import numpy as np
import imageio
import os

from typing import Dict, List, Text, Union
import pickle
import torch
from tqdm import tqdm
from copy import deepcopy
from loguru import logger
import scipy.io as sio

from dannce.engine.data import serve_data_DANNCE, io
from dannce.config import make_paths_safe
from dannce.config import _DEFAULT_VIDDIR

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
                "NOTE: Ignoring {} files numbered above {}".format(extension, maxopt)
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
        exp = load_expdict(params, e, expdict, _DEFAULT_VIDDIR,)

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
                partition["valid_sampleIDs"] = pickle.load(f)
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
            pickle.dump(partition["valid_sampleIDs"], f)

        with open(os.path.join(results_dir, "train_samples.pickle"), "wb") as f:
            pickle.dump(partition["train_sampleIDs"], f)
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
            partition["valid_sampleIDs"] = pickle.load(f)
        partition["train_sampleIDs"] = [
            f for f in samples if f not in partition["valid_sampleIDs"]
        ]
    # Save train/val inds
    with open(os.path.join(results_dir, "val_samples.pickle"), "wb") as f:
        pickle.dump(partition["valid_sampleIDs"], f)

    with open(os.path.join(results_dir, "train_samples.pickle"), "wb") as f:
        pickle.dump(partition["train_sampleIDs"], f)

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
    silhouette=False,
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
