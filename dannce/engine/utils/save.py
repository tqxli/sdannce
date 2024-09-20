from typing import Dict
import os
import numpy as np
import pickle
import yaml
from loguru import logger
import scipy.io as sio
from six.moves import cPickle

from dannce.config import make_none_safe
from dannce.engine.data import serve_data_DANNCE


def save_params_pickle(params: Dict):
    """
    save copy of params as pickle for reproducibility.
    """
    handle = open(os.path.join(params["dannce_train_dir"], "params.pickle"), "wb")
    pickle.dump(params, handle)
    handle.close()

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
    handle.close()


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
