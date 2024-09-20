"""Define routines for reading/structuring input data for DANNCE."""
import os
import pickle
import warnings
from copy import deepcopy

import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import median_filter
from scipy.special import comb

from dannce.engine.data import ops as ops
from dannce.engine.data.io import load_camera_params, load_labels, load_sync


def prepare_data(
    params,
    com_flag=True,
    nanflag=False,
    prediction=False,
    predict_labeled_only=False,
    predict_smoothed_labels=False,
    valid=False,
    support=False,
    downsample=1,
    return_cammat=False,
    return_full2d=False,
):
    """Assemble necessary data structures given a set of config params.

    Given a set of config params, assemble necessary data structures and
    return them -- tailored to center of mass finding
    That is, we are refactoring to get rid of unneeded data structures
    (i.e. data 3d)

    multimode: when this True, we output all 2D markers AND their 2D COM
    """
    if prediction:
        # allow predictions only on labeled frames for easier metric evaluation
        labels = (
            load_labels(params["label3d_file"])
            if predict_labeled_only
            else load_sync(params["label3d_file"])
        )
        nFrames = np.max(labels[0]["data_frame"].shape)

        if predict_smoothed_labels:
            nFrames *= 10

        nKeypoints = params["n_channels_out"]
        if "new_n_channels_out" in params.keys():
            if params["new_n_channels_out"] is not None:
                nKeypoints = params["new_n_channels_out"]

        for i in range(len(labels)):
            labels[i]["data_3d"] = np.zeros((nFrames, 3 * nKeypoints))
            labels[i]["data_2d"] = np.zeros((nFrames, 2 * nKeypoints))
    else:
        # print(params["label3d_file"])
        labels = load_labels(params["label3d_file"])

    samples = np.squeeze(labels[0]["data_sampleID"])
    if predict_smoothed_labels:
        sampleIDs_labeled = np.squeeze(
            load_labels(params["label3d_file"])[0]["data_sampleID"]
        )
        sample_inds = [np.where(samples == samp)[0][0] for samp in sampleIDs_labeled]
        expanded_sample_inds = np.concatenate(
            [np.arange(sample_ind - 5, sample_ind + 5) for sample_ind in sample_inds]
        )
        samples = samples[expanded_sample_inds]

    # load camera parameters
    camera_params = load_camera_params(params["label3d_file"])
    cameras = {name: camera_params[i] for i, name in enumerate(params["camnames"])}

    if "m" in camera_params[0] and not params["mirror"]:
        warnings.warn(
            "found mirror field in camera params, but the network is not set to run in mirror mode"
        )
    elif params["mirror"] and "m" not in camera_params[0]:
        raise Exception(
            "network set to run in mirror mode, but cannot find mirror (m) field in camera params"
        )

    # enable temporal training
    TEMPORAL_FLAG = (not prediction) and (
        params.get("use_temporal", False)
    )  # and (not valid)
    chunk_list = None
    if TEMPORAL_FLAG:
        samples, labels, chunk_list = prepare_temporal_seqs(
            params, samples, labels, downsample, valid, support
        )

    if labels[0]["data_sampleID"].shape == (1, 1):
        # Then the squeezed value is just a number, so we add to to a list so
        # that is can be iterated over downstream
        samples = [samples]
        warnings.warn("Note: only 1 sample in label file")

    # Collect data labels and matched frames info. We will keep the 2d labels
    # here just because we could in theory use this for training later.
    # No need to collect 3d data but it useful for checking predictions
    if len(params["camnames"]) != len(labels):
        raise Exception("need an entry in label3d_file for every camera")

    framedict = {}
    ddict = {}

    for i, label in enumerate(labels):
        framedict[params["camnames"][i]] = np.squeeze(label["data_frame"])
        data = label["data_2d"]

        # reshape data_2d so that it is shape (time points, 2, 20)
        if len(data.shape) == 2:
            data = np.transpose(np.reshape(data, [data.shape[0], -1, 2]), [0, 2, 1])

        # Correct for Matlab "1" indexing
        data = data - 1

        if params["mirror"] and cameras[params["camnames"][i]]["m"] == 1:
            # then we need to flip the 2D coords -- for now assuemd only horizontal flipping
            data[:, 1] = params["raw_im_h"] - data[:, 1] - 1

        if params["multi_mode"]:
            print("Entering multi-mode with {} + 1 targets".format(data.shape[-1]))
            if nanflag:
                dcom = np.mean(data, axis=2, keepdims=True)
            else:
                dcom = np.nanmean(data, axis=2, keepdims=True)
            data = np.concatenate((data, dcom), axis=-1)
        elif (not return_full2d) and com_flag:
            # Convert to COM only if not already
            if len(data.shape) == 3 and params["n_instances"] == 1:
                if nanflag:
                    data = np.mean(data, axis=2)
                else:
                    data = (
                        np.nan * np.ones(data.shape[:2])
                        if np.isnan(data).all()
                        else np.nanmean(data, axis=2)
                    )
                data = data[:, :, np.newaxis]

        ddict[params["camnames"][i]] = data

    data_3d = labels[0]["data_3d"]
    if len(data_3d.shape) == 2:
        data_3d = np.transpose(
            np.reshape(data_3d, [data_3d.shape[0], -1, 3]), [0, 2, 1]
        )

    # If specific markers are set to be excluded, set them to NaN here.
    if params["drop_landmark"] is not None and not prediction:
        print(
            "Setting landmarks {} to NaN. These landmarks will not be included in loss or metric evaluations".format(
                params["drop_landmark"]
            )
        )
        data_3d[:, :, params["drop_landmark"]] = np.nan

    datadict = {}
    datadict_3d = {}
    for i in range(len(samples)):
        frames = {}
        data = {}
        for j in range(len(params["camnames"])):
            frames[params["camnames"][j]] = framedict[params["camnames"][j]][i]
            data[params["camnames"][j]] = ddict[params["camnames"][j]][i]
        datadict[samples[i]] = {"data": data, "frames": frames}
        datadict_3d[samples[i]] = data_3d[i]

    if return_cammat:
        camera_mats = {
            name: ops.camera_matrix(cam["K"], cam["r"], cam["t"])
            for name, cam in cameras.items()
        }
        return samples, datadict, datadict_3d, cameras, camera_mats

    return samples, datadict, datadict_3d, cameras, chunk_list


def get_seq_bounds(seqlen):
    left_bound = -int(seqlen // 2)
    right_bound = (
        int(np.round(seqlen / 2)) if seqlen % 2 == 0 else int(np.round(seqlen / 2)) + 1
    )
    return left_bound, right_bound


def get_chunks(sample_inds, left_bound, right_bound, maxlen, downsample):
    chunk_ind_list = []
    for sample_ind in sample_inds:
        # check chunk validity
        l_lim = sample_ind + left_bound * downsample
        r_lim = sample_ind + right_bound * downsample
        shift = 0

        if l_lim < 0:
            shift = np.ceil(np.abs(l_lim) / downsample) * downsample
        elif r_lim >= maxlen:
            shift = -(np.ceil((r_lim + 1 - maxlen) / downsample) - 1) * downsample

        chunk = np.arange(l_lim, r_lim, downsample) + shift

        chunk_ind_list.append(chunk)

    all_samples_inds = np.concatenate(chunk_ind_list).astype(int)
    return all_samples_inds


def prepare_temporal_seqs(
    params, samples, labels, downsample=1, valid=False, support=False
):
    """
    For temporal training, prepare samples in form of consecutive chunks.
    """
    assert params[
        "temporal_chunk_size"
    ], "PLease specify the temporal sequence size for temporal loss/training."

    temp_n = params["temporal_chunk_size"]

    # load in all sampleIDs
    labels_extra = load_sync(params["label3d_file"])
    samples_extra = np.squeeze(labels_extra[0]["data_sampleID"])

    # select extra samples from the neighborhood of labeled samples
    # each of which is referred as a "temporal chunk"
    left_bound, right_bound = get_seq_bounds(temp_n)

    # what if we want to use the unlabeled frames in the test set for pretraining
    sample_inds, samples_inds_unlabeled, samples_test_inds = [], [], []
    if (support) and isinstance(params["n_support_chunks"], int):
        samples_test_inds = np.random.choice(
            samples_extra[-left_bound::temp_n],
            size=params["n_support_chunks"],
            replace=False,
        )
        samples_test_inds = sorted(list(samples_test_inds))
        print(
            "For unsupervised training, load in {} unlabeled chunks from the valid/test recording.".format(
                params["n_support_chunks"]
            )
        )
        samples = None
    else:
        # locate labeled frames
        sample_inds = [np.where(samples_extra == samp)[0][0] for samp in samples]

        # if specified, load in extra completely unlabaled chunks
        if (params["unlabeled_temp"] > 0) and (not valid):
            all_samples_inds_unlabeled = np.array(
                list(set(np.arange(len(samples_extra))) - set(sample_inds))
            )
            # n_unlabeled_temp = int(params["unlabeled_temp"])
            n_unlabeled_temp = int(np.ceil(len(samples) * params["unlabeled_temp"]))
            print(
                "Load in {} unlabeled temporal chunks, in addition to {} labels.".format(
                    n_unlabeled_temp, len(samples)
                )
            )
            samples_inds_unlabeled = np.random.choice(
                all_samples_inds_unlabeled, size=n_unlabeled_temp, replace=False
            )
            samples_inds_unlabeled = sorted(list(samples_inds_unlabeled))

    sample_inds = sample_inds + samples_inds_unlabeled + samples_test_inds
    sample_inds = np.array(sample_inds)

    # generate chunks
    all_samples_inds = get_chunks(
        sample_inds, left_bound, right_bound, len(samples_extra), downsample
    )

    # there can be repetitive sampleIDs,
    # we only load in each label once in datadict and datadict3d
    # during training, the dataset will fetch the correct chunk using chunk_list
    all_samples = samples_extra[all_samples_inds]
    chunk_list = [
        all_samples[i : i + temp_n] for i in range(0, len(all_samples), temp_n)
    ]
    all_samples, unique_index = np.unique(all_samples, return_index=True)
    labeled_inds = (
        np.array([np.where(all_samples == samp)[0][0] for samp in samples])
        if samples is not None
        else None
    )

    samples = all_samples

    for i, label in enumerate(labels):
        for k in ["data_frame", "data_2d", "data_3d"]:
            if label[k].shape[0] == 1:
                label[k] = label[k].T
            if labels_extra[i][k].shape[0] == 1:
                labels_extra[i][k] = labels_extra[i][k].T

            if k == "data_frame":
                label[k] = labels_extra[i][k][all_samples_inds[unique_index]]
            else:
                temp_data = np.nan * np.ones((len(samples), label[k].shape[1]))
                if labeled_inds is not None:
                    temp_data[labeled_inds] = label[k]
                label[k] = temp_data

    return samples, labels, chunk_list


def prepare_COM_multi_instance(
    comfile,
    datadict,
    comthresh=0.0,
    weighted=False,
    camera_mats=None,
    conf_rescale=None,
    linking_method="euclidean",
):
    """Replace 2d coords with preprocessed COM coords, return 3d COM coords.

    Loads COM file, replaces 2D coordinates in datadict with the preprocessed
    COM coordinates, returns dict of 3d COM coordinates

    Thresholds COM predictions at comthresh w.r.t. saved pred_max values.
    Averages only the 3d coords for camera pairs that both meet thresh.
    Returns nan for 2d COM if camera does not reach thresh. This should be
    detected by the generator to return nans such that bad camera
    frames do not get averaged in to image data
    """

    with open(comfile, "rb") as f:
        com = pickle.load(f)
    com3d_dict = {}

    firstkey = list(com.keys())[0]

    camnames = np.array(list(datadict[list(datadict.keys())[0]]["data"].keys()))

    # Because I repeat cameras to fill up 6 camera quota, I need grab only
    # the unique names
    _, idx = np.unique(camnames, return_index=True)
    uCamnames = camnames[np.sort(idx)]

    # It's possible that the keys in the COM dict are strings with an experiment ID
    # prepended in front. We need to handle this appropriately.
    if isinstance(firstkey, str):
        com_ = {}
        for key in com.keys():
            com_[int(float(key.split("_")[-1]))] = com[key]
        com = com_

    fcom = list(com.keys())[0]

    # Grab the multi-instance predictions and store in single matrix
    coms = [v["triangulation"]["instances"] for v in com.values()]
    coms = [np.concatenate(v, axis=1) for v in coms]
    coms = np.stack(coms, axis=2).transpose([2, 0, 1])

    if linking_method == "euclidean":
        # Use a 1-frame euclidean distance metric to string together identities.
        # Currently just for 2 instances
        for n_sample in range(1, coms.shape[0]):
            same_dist1 = np.sqrt(
                np.sum((coms[n_sample, :, 0] - coms[n_sample - 1, :, 0]) ** 2)
            )
            diff_dist1 = np.sqrt(
                np.sum((coms[n_sample, :, 0] - coms[n_sample - 1, :, 1]) ** 2)
            )
            same_dist2 = np.sqrt(
                np.sum((coms[n_sample, :, 1] - coms[n_sample - 1, :, 1]) ** 2)
            )
            diff_dist2 = np.sqrt(
                np.sum((coms[n_sample, :, 1] - coms[n_sample - 1, :, 0]) ** 2)
            )
            same = np.mean([same_dist1, same_dist2])
            diff = np.mean([diff_dist1, diff_dist2])
            if diff < same:
                temp = coms[n_sample, :, 0].copy()
                coms[n_sample, :, 0] = coms[n_sample, :, 1]
                coms[n_sample, :, 1] = temp
    elif linking_method == "kalman":
        pass
    elif linking_method == "multi_channel":
        a = []
    else:
        raise Exception("Invalid linking method.")

    # Return to com3d_dict format.
    for i, key in enumerate(com.keys()):
        com3d_dict[key] = coms[i, :, :]

    return None, com3d_dict


def prepare_COM(
    comfile,
    datadict,
    comthresh=0.0,
    weighted=False,
    camera_mats=None,
    conf_rescale=None,
    method="median",
):
    """Replace 2d coords with preprocessed COM coords, return 3d COM coords.

    Loads COM file, replaces 2D coordinates in datadict with the preprocessed
    COM coordinates, returns dict of 3d COM coordinates

    Thresholds COM predictions at comthresh w.r.t. saved pred_max values.
    Averages only the 3d coords for camera pairs that both meet thresh.
    Returns nan for 2d COM if camera does not reach thresh. This should be
    detected by the generator to return nans such that bad camera
    frames do not get averaged in to image data
    """

    with open(comfile, "rb") as f:
        com = pickle.load(f)
    com3d_dict = {}

    if method == "mean":
        print("using mean to get 3D COM")

    elif method == "median":
        print("using median to get 3D COM")

    firstkey = list(com.keys())[0]

    camnames = np.array(list(datadict[list(datadict.keys())[0]]["data"].keys()))

    # Because I repeat cameras to fill up 6 camera quota, I need grab only
    # the unique names
    _, idx = np.unique(camnames, return_index=True)
    uCamnames = camnames[np.sort(idx)]

    # It's possible that the keys in the COM dict are strings with an experiment ID
    # prepended in front. We need to handle this appropriately.
    if isinstance(firstkey, str):
        com_ = {}
        for key in com.keys():
            com_[int(float(key.split("_")[-1]))] = com[key]
        com = com_

    fcom = list(com.keys())[0]
    for key in com.keys():
        this_com = com[key]

        if key in datadict.keys():
            for k in range(len(camnames)):
                datadict[key]["data"][camnames[k]] = this_com[camnames[k]]["COM"][
                    :, np.newaxis
                ].astype("float32")

                # Quick & dirty way to dynamically scale the confidence map output
                if conf_rescale is not None and camnames[k] in conf_rescale.keys():
                    this_com[camnames[k]]["pred_max"] *= conf_rescale[camnames[k]]

                # then, set to nan
                if this_com[camnames[k]]["pred_max"] <= comthresh:
                    datadict[key]["data"][camnames[k]][:] = np.nan

            com3d = np.zeros((3, int(comb(len(uCamnames), 2)))) * np.nan
            weights = np.zeros((int(comb(len(uCamnames), 2)),))
            cnt = 0
            for j in range(len(uCamnames)):
                for k in range(j + 1, len(uCamnames)):
                    if (this_com[uCamnames[j]]["pred_max"] > comthresh) and (
                        this_com[uCamnames[k]]["pred_max"] > comthresh
                    ):
                        if (
                            "{}_{}".format(uCamnames[j], uCamnames[k])
                            in this_com["triangulation"].keys()
                        ):
                            com3d[:, cnt] = this_com["triangulation"][
                                "{}_{}".format(uCamnames[j], uCamnames[k])
                            ]
                        elif (
                            "{}_{}".format(uCamnames[k], uCamnames[j])
                            in this_com["triangulation"].keys()
                        ):
                            com3d[:, cnt] = this_com["triangulation"][
                                "{}_{}".format(uCamnames[k], uCamnames[j])
                            ]
                        else:
                            raise Exception(
                                "Could not find this camera pair: {}".format(
                                    "{}_{}".format(uCamnames[k], uCamnames[j])
                                )
                            )
                        weights[cnt] = (
                            this_com[uCamnames[j]]["pred_max"]
                            * this_com[uCamnames[k]]["pred_max"]
                        )
                    cnt += 1

            # weigts produces a weighted average of COM based on our overall confidence
            if weighted:
                if np.sum(weights) != 0:
                    weights = weights / np.sum(weights)
                    com3d = np.nansum(com3d * weights[np.newaxis, :], axis=1)
                else:
                    com3d = np.zeros((3,)) * np.nan
            else:
                if method == "mean":
                    com3d = np.nanmean(com3d, axis=1)
                elif method == "median":
                    com3d = np.nanmedian(com3d, axis=1)
                else:
                    raise Exception("Uknown 3D COM method")

            com3d_dict[key] = com3d
        else:
            warnings.warn("Key in COM file but not in datadict")

    return datadict, com3d_dict


def prepare_com3ddict(datadict_3d):
    """Take the mean of the 3d data.

    Call this when using ground truth 3d anchor points that do not need to be
    loaded in via a special com file -- just need to take the mean
    of the 3d data with the 3d datadict
    """
    com3d_dict = {}
    for key in datadict_3d.keys():
        com3d_dict[key] = np.nanmean(datadict_3d[key], axis=-1)
    return com3d_dict


def addCOM(d3d_dict, c3d_dict):
    """Add COM back in to data.

    For JDM37 data and its ilk, the data are loaded in centered to the
    animal center of mass (Because they were predictions from the network)
    We need to add the COM back in, because durign training everything gets
    centered to the true COM again
    """
    for key in c3d_dict.keys():
        d3d_dict[key] = d3d_dict[key] + c3d_dict[key][:, np.newaxis]
    return d3d_dict


def remove_samples(s, d3d, mode="clean", auxmode=None):
    """Filter data structures for samples that meet inclusion criteria (mode).

    mode == 'clean' means only use samples in which all ground truth markers
             are recorded
    mode == 'SpineM' means only remove data where SpineM is missing
    mode == 'liberal' means include any data that isn't *all* nan
    aucmode == 'JDM52d2' removes a really bad marker period -- samples 20k to 32k
    I need to cull the samples array (as this is used to index eveyrthing else),
    but also the
    data_3d_ array that is used to for finding clusters
    """
    sample_mask = np.ones((len(s),), dtype="bool")

    if mode == "clean":
        for i in range(len(s)):
            if np.isnan(np.sum(d3d[i])):
                sample_mask[i] = 0
    elif mode == "liberal":
        for i in range(len(s)):
            if np.all(np.isnan(d3d[i])):
                sample_mask[i] = 0

    if auxmode == "JDM52d2":
        print("removing bad JDM52d2 frames")
        for i in range(len(s)):
            if s[i] >= 20000 and s[i] <= 32000:
                sample_mask[i] = 0

    s = s[sample_mask]
    d3d = d3d[sample_mask]

    # zero the 3d data to SpineM
    d3d[:, ::3] -= d3d[:, 12:13]
    d3d[:, 1::3] -= d3d[:, 13:14]
    d3d[:, 2::3] -= d3d[:, 14:15]
    return s, d3d


def remove_samples_com(s, com3d_dict, cthresh=350, rmc=False):
    """Remove any remaining samples in which the 3D COM estimates are nan.

    (i.e. no camera pair above threshold for a given frame)
    Also, let's remove any sample where abs(COM) is > 350
    """
    sample_mask = np.ones((len(s),), dtype="bool")

    for i in range(len(s)):
        if s[i] not in com3d_dict:
            sample_mask[i] = 0
        else:
            if np.isnan(np.sum(com3d_dict[s[i]])):
                sample_mask[i] = 0
            if rmc:
                if np.any(np.abs(com3d_dict[s[i]]) > cthresh):
                    sample_mask[i] = 0

    s = s[sample_mask]
    return s


def add_experiment(
    experiment,
    samples_out,
    datadict_out,
    datadict_3d_out,
    com3d_dict_out,
    samples_in,
    datadict_in,
    datadict_3d_in,
    com3d_dict_in,
    temporal_chunks_out=None,
    temporal_chunks_in=None,
):
    samples_in = [str(experiment) + "_" + str(int(x)) for x in samples_in]
    samples_out = samples_out + samples_in

    if temporal_chunks_in is not None:
        for chunk in temporal_chunks_in:
            if experiment not in temporal_chunks_out.keys():
                temporal_chunks_out[experiment] = []
            temporal_chunks_out[experiment].append(
                np.array([str(experiment) + "_" + str(s) for s in chunk])
            )

    for key in datadict_in.keys():
        datadict_out[str(experiment) + "_" + str(int(key))] = datadict_in[key]

    for key in datadict_3d_in.keys():
        datadict_3d_out[str(experiment) + "_" + str(int(key))] = datadict_3d_in[key]

    for key in com3d_dict_in.keys():
        com3d_dict_out[str(experiment) + "_" + str(int(key))] = com3d_dict_in[key]

    return (
        samples_out,
        datadict_out,
        datadict_3d_out,
        com3d_dict_out,
        temporal_chunks_out,
    )


def prepend_experiment(
    params, datadict, num_experiments, camnames, cameras, dannce_prediction=False,
):
    """
    Adds necessary experiment labels to data structures. E.g. experiment 0 CameraE's "camname"
        Becomes 0_CameraE.
    """
    cameras_ = {}
    datadict_ = {}
    new_chunks = {}
    prev_camnames = camnames.copy()
    for e in range(num_experiments):

        # Create a unique camname for each camera in each experiment
        cameras_[e] = {}
        for key in cameras[e]:
            cameras_[e][str(e) + "_" + key] = cameras[e][key]

        camnames[e] = [str(e) + "_" + f for f in camnames[e]]
        params["experiment"][e]["camnames"] = camnames[e]

        for n_cam, name in enumerate(camnames[e]):
            # print(name)
            # print(params["experiment"][e]["chunks"][name])
            if dannce_prediction:
                try:
                    new_chunks[name] = params["experiment"][e]["chunks"][
                        prev_camnames[e][n_cam]
                    ]
                except:
                    new_chunks[name] = params["experiment"][e]["chunks"][name]
            else:
                new_chunks[name] = params["experiment"][e]["chunks"][name]
        params["experiment"][e]["chunks"] = new_chunks

    for key in datadict.keys():
        enum = key.split("_")[0]
        datadict_[key] = {}
        datadict_[key]["data"] = {}
        datadict_[key]["frames"] = {}
        for key_ in datadict[key]["data"]:
            datadict_[key]["data"][enum + "_" + key_] = datadict[key]["data"][key_]
            datadict_[key]["frames"][enum + "_" + key_] = datadict[key]["frames"][key_]

    return cameras_, datadict_, params


def identify_exp_pairs(exps):
    """For multi-instance social behaviorial dannce, 
       identify social animal pairs from all the exps.

       One example would be 
       '.../dannce_rig/ratsInColor/2021_07_07_M1_M6/20210813_175716_Label3D_B_dannce.mat'
       and 
       '.../dannce_rig/ratsInColor/2021_07_07_M1_M6/20210813_195437_Label3D_R_dannce.mat'
       each corresponding to one of the animals present in the same scene.
    args: 
        exps: Dict. Keys are integers [0, n_exps]. 
              Each value is a Dict containing single experiment information
    return:
        pair_list: List of tuples of indices
        [[1, 3], [0, 5, 10], ...]
    """
    exp_indices = sorted(exps.keys())
    exp_base_folders = np.array([exps[i]["base_exp_folders"] for i in exp_indices])

    # use np.unique to identify exps with the same base_exp_folders
    uniques, counts = np.unique(exp_base_folders, return_counts=True)
    pair_indices = counts >= 2

    # find all pairs
    pairs = []
    for i in pair_indices:
        pairs.append(np.where(exp_base_folders == uniques[i]))

    return pairs


def collate_fn(items):
    volumes = torch.cat([item[0] for item in items], dim=0)  # .permute(0, 4, 1, 2, 3)
    targets = torch.cat([item[2] for item in items], dim=0)

    try:
        grids = torch.cat([item[1] for item in items], dim=0)
    except:
        grids = None

    try:
        auxs = torch.cat([item[3] for item in items], dim=0)
    except:
        auxs = None

    return volumes, grids, targets, auxs


def setup_dataloaders(train_dataset, valid_dataset, params):
    # current implementation returns chunked data
    if params["use_temporal"]:
        valid_batch_size = params["batch_size"] // params["temporal_chunk_size"]
    # elif params["is_social_dataset"]:
    #     valid_batch_size = params["batch_size"] // 2
    else:
        valid_batch_size = params["batch_size"]

    if params["multi_gpu_train"] and len(params["gpu_id"]) > 1:
        valid_batch_size = valid_batch_size * len(params["gpu_id"])
        print(f"Use batch size of {valid_batch_size} for multiple GPUs.")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=valid_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
        persistent_workers=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        persistent_workers=True,
    )
    return train_dataloader, valid_dataloader


NPY_DIRNAMES = ["image_volumes", "grid_volumes", "targets"]
NPY_SOCIAL_DIRNAMES = ["occlusion_scores"]
AUX_NPY_DIRNAMES = ["visual_hulls"]


def examine_npy_training(params, samples, aux=False):
    TO_BE_EXAMINED = AUX_NPY_DIRNAMES if aux else NPY_DIRNAMES
    if params["is_social_dataset"] and params["downscale_occluded_view"]:
        TO_BE_EXAMINED = TO_BE_EXAMINED + NPY_SOCIAL_DIRNAMES

    npydir, missing_npydir = {}, {}

    for e in range(len(params["exp"])):
        # for social, cannot use the same default npy volume dir for both animals
        label3d_name = os.path.basename(params["experiment"][e]["label3d_file"]).split(
            ".mat"
        )[0]
        npy_folder = (
            params["experiment"][e]["npy_vol_dir"]
            + "/"
            + str(params["nvox"])
            + "-"
            + label3d_name
        )
        npydir[e] = npy_folder

        # create missing npy directories
        if not os.path.exists(npydir[e]):
            missing_npydir[e] = npydir[e]
            for dir in TO_BE_EXAMINED:
                os.makedirs(os.path.join(npydir[e], dir))
        else:
            for dir in TO_BE_EXAMINED:
                dirpath = os.path.join(npydir[e], dir)
                if (not os.path.exists(dirpath)) or (len(os.listdir(dirpath)) == 0):
                    missing_npydir[e] = npydir[e]
                    os.makedirs(dirpath, exist_ok=True)

    missing_samples = [
        samp
        for samp in samples
        if int(samp.split("_")[0]) in list(missing_npydir.keys())
    ]

    # check any other missing npy samples
    for samp in list(set(samples) - set(missing_samples)):
        e, sampleID = int(samp.split("_")[0]), samp.split("_")[1]
        if not os.path.exists(
            os.path.join(npydir[e], "image_volumes", f"0_{sampleID}.npy")
        ):
            missing_samples.append(samp)
            missing_npydir[e] = npydir[e]

    missing_samples = np.array(sorted(missing_samples))

    return npydir, missing_npydir, missing_samples
