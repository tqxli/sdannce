import numpy as np
import scipy.io as sio
import mat73

# helper functions
def project_to_2d(
    pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3d points to 2d.
    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """

    M = np.concatenate((R, t), axis=0) @ K
    projPts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) @ M
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:]

    return projPts


def distortPoints(points, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """Distort points according to camera parameters.
    Ported from Matlab 2018a
    """
    # unpack the intrinisc matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = np.array([cx, cy])
    centeredPoints = points - center[np.newaxis, :]

    # normalize the points
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm ** 2 + yNorm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2]
    if len(radialDistortion) < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = np.stack((xNorm, yNorm)).T
    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * np.array([alpha, alpha]).T
        + np.stack((dxTangential, dyTangential)).T
    )

    # # convert back to pixels
    distortedPointsX = (
        (distortedNormalizedPoints[:, 0] * fx)
        + cx
        + (skew * distortedNormalizedPoints[:, 1])
    )
    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy
    distortedPoints = np.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


def load_cameras(path):
    mat73_flag = False
    # breakpoint()
    try:
        d = sio.loadmat(path)
        # camnames = [cam[0][0] for cam in d['camnames']]
        camnames = [cam[0] for cam in d["camnames"][0]]
    except:
        d = mat73.loadmat(path)
        camnames = [name[0] for name in d["camnames"]]
        mat73_flag = True
    fns = ["K", "RDistort", "TDistort", "r", "t"]

    # breakpoint()
    # camnames = [cam[0] for cam in d['camnames'][0]]
    cam_params = d["params"]
    cameras = {}
    for i, camname in enumerate(camnames):
        cameras[camname] = {}
        for j, fn in enumerate(fns):
            if mat73_flag:
                cameras[camname][fn] = cam_params[i][0][fn]
            else:
                cameras[camname][fn] = cam_params[i][0][0][0][j]
        if len(cameras[camname]["t"].shape) == 1:
            cameras[camname]["t"] = cameras[camname]["t"][np.newaxis, ...]
    return cameras


def load_label3d_data(path, key):
    """Load Label3D data

    Args:
        path (Text): Path to Label3D file
        key (Text): Field to access

    Returns:
        TYPE: Data from field
    """
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
            for key in d.dtype.names[:4]:
                # breakpoint()
                d_[key] = d[key][0, 0]
                # print(key, d_[key].shape)
            data.append(d_)
    except:
        d = mat73.loadmat(path)[key]
        data = [f[0] for f in d]
    return data


def load_sync(path):
    """Load synchronization data from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of synchronization dictionaries.
    """
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path):
    """Load labelData from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of labelData dictionaries.
    """
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path):
    """Load COM from .mat file.

    Args:
        path (Text): Path to .mat file with "com" field

    Returns:
        Dict: Dictionary with com data
    """
    try:
        d = sio.loadmat(path)["com"]
        # data = {}
        data = d["com3d"][0, 0]
        # data["sampleID"] = d["sampleID"][0, 0].astype(int)
    except:
        data = mat73.loadmat(path)["com"]["com3d"]
        # breakpoint()
        # data["sampleID"] = data["sampleID"].astype(int)
    return data
