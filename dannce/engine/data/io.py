"""Label3D data loading and saving operations."""
from typing import Dict, List, Text, Union

import mat73
import numpy as np
import scipy.io as sio


def load_label3d_data(path: Text, key: Text):
    """Load Label3D data

    Args:
        path (Text): Path to Label3D file
        key (Text): Field to access

    Returns:
        TYPE: Data from field
    """
    if path.endswith(".npz"):
        raise NotImplementedError(path)
    try:
        d = sio.loadmat(path)[key]
        dataset = [f[0] for f in d]

        # Data are loaded in this annoying structure where the array
        # we want is at dataset[i][key][0,0], as a nested array of arrays.
        # Simplify this structure (a numpy record array) here.
        # Additionally, cannot use views here because of shape mismatches. Define
        # new dict and return.
        data = []
        for d in dataset:
            d_ = {}
            for key in d.dtype.names:
                d_[key] = d[key][0, 0]
            data.append(d_)
    except:
        d = mat73.loadmat(path)[key]
        data = [f[0] for f in d]
    return data


def load_camera_params(path: Text) -> List[Dict]:
    """Load camera parameters from Label3D file.

    Args:
        path (Text): Path to Label3D file

    Returns:
        List[Dict]: List of camera parameter dictionaries.
    """
    if path.endswith('.npz'):
        d = np.load(path)
        return [dict(
            r=d['r'][i].T,
            R=d['r'][i].T, # Note case mismatch
            t=d['t'][i].reshape(1,3),
            K=d['K'][i].T,
            RDistort=d['RDistort'][i].reshape(1,-1),
            TDistort=d['TDistort'][i].reshape(1,-1),
        ) for i in range(len(d['r']))]
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
        if len(p["t"].shape) == 1:
            p["t"] = p["t"][np.newaxis, ...]
    return params


def load_sync(path: Text) -> List[Dict]:
    """Load synchronization data from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of synchronization dictionaries.
    """
    if path.endswith('.npz'):
        C = len(load_camnames(path))
        # d = dict(np.load(path))

        # data_frame = d['data_frame'].astype(int)
        # data_sampleID = d['data_sampleID'].astype(int)

        data_frame = np.arange(30000, dtype=int) # TEMP!
        data_sampleID = data_frame # TMEP!
       
        out = []
        for i in range(C):
            out.append(dict(
                data_frame=data_frame,
                data_sampleID=data_sampleID,
            ))
        return out
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path: Text) -> List[Dict]:
    """Load labelData from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of labelData dictionaries.
    """
    if path.endswith('.npz'):
        C = len(load_camnames(path))
        d = dict(np.load(path))

        data_frame = d['data_frame'].astype(int)
        data_sampleID = d['data_sampleID'].astype(int)

        data_2d = d['data_2d'] # (T,K,C,2)
        data_3d = d['data_3d'] # (T,K,3)

        out = []
        for i in range(C):
            out.append(dict(
                data_frame=data_frame,
                data_sampleID=data_sampleID,
                data_2d=np.transpose(data_2d[:,:,i], (0,2,1)),
                data_3d=np.transpose(data_3d, (0,2,1)),
            ))
        return out
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path: Text) -> Dict:
    """Load COM from .mat file.

    Args:
        path (Text): Path to .mat file with "com" field

    Returns:
        Dict: Dictionary with com data
    """
    if path.endswith(".npz"):
        raise NotImplementedError(path)
    try:
        d = sio.loadmat(path)["com"]
        data = {}
        data["com3d"] = d["com3d"][0, 0]
        data["sampleID"] = d["sampleID"][0, 0].astype(int)
    except:
        data = mat73.loadmat(path)["com"]
        data["sampleID"] = data["sampleID"].astype(int)
    return data


def load_camnames(path: Text) -> Union[List, None]:
    """Load camera names from .mat file.

    Args:
        path (Text): Path to .mat file with "camnames" field

    Returns:
        Union[List, None]: List of cameranames
    """
    if path.endswith(".npz"):
        return np.load(path)["camnames"]
    try:
        label_3d_file = sio.loadmat(path)
        if "camnames" in label_3d_file:
            names = label_3d_file["camnames"][:]
            if len(names) != len(label_3d_file["labelData"]):
                camnames = [name[0] for name in names[0]]
            else:
                camnames = [name[0][0] for name in names]
        else:
            camnames = None
    except:
        label_3d_file = mat73.loadmat(path)
        if "camnames" in label_3d_file:
            camnames = [name[0] for name in label_3d_file["camnames"]]
    return camnames
