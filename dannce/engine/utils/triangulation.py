from typing import Dict, Text
import numpy as np

from dannce.engine.data import ops
import dannce.engine.utils.image as image_utils
from dannce.engine.utils.debug import debug_com


def extract_multi_instance_single_channel(
    pred: np.ndarray,
    pred_batch: np.ndarray,
    n_cam: int,
    sample_id: Text,
    n_frame: int,
    n_batch: int,
    params: Dict,
    save_data: Dict,
    cameras: Dict,
    generator,
) -> Dict:
    """Extract prediction indices for multi-instance single-channel tracking.

    Args:
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        n_cam (int): Camera number
        sample_id (Text): Sample identifier
        n_frame (int): Frame number
        n_batch (int): Batch number
        params (Dict): Parameters dictionary.
        save_data (Dict): Saved data dictionary.
        cameras (Dict): Camera dictionary
        generator (keras.utils.Sequence): DataGenerator

    No Longer Returned:
        (Dict): Updated saved data dictionary.
    """
    pred_max = np.max(np.squeeze(pred[n_cam]))
    ind = (
        np.array(
            image_utils.get_peak_inds_multi_instance(
                np.squeeze(pred[n_cam]), params["n_instances"], window_size=3,
            )
        )
        * params["downfac"]
    )
    for instance in range(params["n_instances"]):
        ind[instance, 0] += params["crop_height"][0]
        ind[instance, 1] += params["crop_width"][0]
        ind[instance, :] = ind[instance, ::-1]

    # now, the center of mass is (x,y) instead of (i,j)
    # now, we need to use camera calibration to triangulate
    # from 2D to 3D
    if params["com_debug"] is not None:
        cnum = params["camnames"].index(params["com_debug"])
        if n_cam == cnum:
            debug_com(
                params, pred, pred_batch, generator, ind, n_frame, n_batch, n_cam,
            )

    save_data[sample_id][params["camnames"][n_cam]] = {
        "pred_max": pred_max,
        "COM": ind,
    }

    # Undistort this COM here.
    for instance in range(params["n_instances"]):
        pts1 = np.squeeze(
            save_data[sample_id][params["camnames"][n_cam]]["COM"][instance, :]
        )
        pts1 = pts1[np.newaxis, :]
        pts1 = ops.unDistortPoints(
            pts1,
            cameras[params["camnames"][n_cam]]["K"],
            cameras[params["camnames"][n_cam]]["RDistort"],
            cameras[params["camnames"][n_cam]]["TDistort"],
            cameras[params["camnames"][n_cam]]["R"],
            cameras[params["camnames"][n_cam]]["t"],
        )
        save_data[sample_id][params["camnames"][n_cam]]["COM"][
            instance, :
        ] = np.squeeze(pts1)

    return save_data


def extract_multi_instance_multi_channel(
    pred: np.ndarray,
    pred_batch: np.ndarray,
    n_cam: int,
    sample_id: Text,
    n_frame: int,
    n_batch: int,
    params: Dict,
    save_data: Dict,
    cameras: Dict,
    generator,
) -> Dict:
    """Extract prediction indices for multi-instance multi-channel tracking.

    Args:
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        n_cam (int): Camera number
        sample_id (Text): Sample identifier
        n_frame (int): Frame number
        n_batch (int): Batch number
        params (Dict): Parameters dictionary.
        save_data (Dict): Saved data dictionary.
        cameras (Dict): Camera dictionary
        generator (keras.utils.Sequence): DataGenerator

    No Longer Returned:
        (Dict): Updated saved data dictionary.
    """
    save_data[sample_id][params["camnames"][n_cam]] = {
        "COM": np.zeros((params["n_instances"], 2)),
    }
    for instance in range(params["n_instances"]):
        pred_max = np.max(np.squeeze(pred[n_cam, :, :, instance]))
        ind = (
            np.array(image_utils.get_peak_inds(np.squeeze(pred[n_cam, :, :, instance])))
            * params["downfac"]
        )
        ind[0] += params["crop_height"][0]
        ind[1] += params["crop_width"][0]
        ind = ind[::-1]
        # now, the center of mass is (x,y) instead of (i,j)
        # now, we need to use camera calibration to triangulate
        # from 2D to 3D
        if params["com_debug"] is not None:
            cnum = params["camnames"].index(params["com_debug"])
            if n_cam == cnum:
                debug_com(
                    params,
                    pred,
                    pred_batch,
                    generator,
                    ind,
                    n_frame,
                    n_batch,
                    n_cam,
                    instance,
                )

        # Undistort this COM here.
        pts = np.squeeze(ind)
        pts = pts[np.newaxis, :]
        pts = ops.unDistortPoints(
            pts,
            cameras[params["camnames"][n_cam]]["K"],
            cameras[params["camnames"][n_cam]]["RDistort"],
            cameras[params["camnames"][n_cam]]["TDistort"],
            cameras[params["camnames"][n_cam]]["R"],
            cameras[params["camnames"][n_cam]]["t"],
        )
        save_data[sample_id][params["camnames"][n_cam]]["COM"][
            instance, :
        ] = np.squeeze(pts)

        # TODO(pred_max): Currently only saves for one instance.
        save_data[sample_id][params["camnames"][n_cam]]["pred_max"] = pred_max
    return save_data


def extract_single_instance(
    pred: np.ndarray,
    pred_batch: np.ndarray,
    n_cam: int,
    sample_id: Text,
    n_frame: int,
    n_batch: int,
    params: Dict,
    save_data: Dict,
    cameras: Dict,
    generator,
):
    """Extract prediction indices for single-instance tracking.

    Args:
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        n_cam (int): Camera number
        sample_id (Text): Sample identifier
        n_frame (int): Frame number
        n_batch (int): Batch number
        params (Dict): Parameters dictionary.
        save_data (Dict): Saved data dictionary.
        cameras (Dict): Camera dictionary
        generator (keras.utils.Sequence): DataGenerator

    No Longer Returned:
        (Dict): Updated saved data dictionary.
    """
    pred_max = np.max(np.squeeze(pred[n_cam]))
    ind = (
        np.array(image_utils.get_peak_inds(np.squeeze(pred[n_cam]))) * params["downfac"]
    )
    ind[0] += params["crop_height"][0]
    ind[1] += params["crop_width"][0]
    ind = ind[::-1]

    # mirror flip each coord if indicated
    if params["mirror"] and cameras[params["camnames"][n_cam]]["m"] == 1:
        ind[1] = params["raw_im_h"] - ind[1] - 1

    # now, the center of mass is (x,y) instead of (i,j)
    # now, we need to use camera calibration to triangulate
    # from 2D to 3D
    if params["com_debug"] is not None:
        cnum = params["camnames"].index(params["com_debug"])
        if n_cam == cnum:
            debug_com(
                params, pred, pred_batch, generator, ind, n_frame, n_batch, n_cam,
            )

    save_data[sample_id][params["camnames"][n_cam]] = {
        "pred_max": pred_max,
        "COM": ind,
    }

    # Undistort this COM here.
    pts1 = save_data[sample_id][params["camnames"][n_cam]]["COM"]
    pts1 = pts1[np.newaxis, :]
    pts1 = ops.unDistortPoints(
        pts1,
        cameras[params["camnames"][n_cam]]["K"],
        cameras[params["camnames"][n_cam]]["RDistort"],
        cameras[params["camnames"][n_cam]]["TDistort"],
        cameras[params["camnames"][n_cam]]["R"],
        cameras[params["camnames"][n_cam]]["t"],
    )
    save_data[sample_id][params["camnames"][n_cam]]["COM"] = np.squeeze(pts1)
    return save_data


def triangulate_single_instance(
    n_cams: int, sample_id: Text, params: Dict, camera_mats: Dict, save_data: Dict
) -> Dict:
    """Triangulate for a single instance.

    Args:
        n_cams (int): Numver of cameras
        sample_id (Text): Sample identifier.
        params (Dict): Parameters dictionary.
        camera_mats (Dict): Camera matrices dictioanry.
        save_data (Dict): Saved data dictionary.

    No Longer Returned:
        Dict: Updated saved data dictionary.
    """
    # Triangulate for all unique pairs
    for n_cam1 in range(n_cams):
        for n_cam2 in range(n_cam1 + 1, n_cams):
            pts1 = save_data[sample_id][params["camnames"][n_cam1]]["COM"]
            pts2 = save_data[sample_id][params["camnames"][n_cam2]]["COM"]
            pts1 = pts1[np.newaxis, :]
            pts2 = pts2[np.newaxis, :]

            test3d = ops.triangulate(
                pts1,
                pts2,
                camera_mats[params["camnames"][n_cam1]],
                camera_mats[params["camnames"][n_cam2]],
            ).squeeze()

            save_data[sample_id]["triangulation"][
                "{}_{}".format(params["camnames"][n_cam1], params["camnames"][n_cam2])
            ] = test3d
    return save_data


def triangulate_multi_instance_multi_channel(
    n_cams: int, sample_id: Text, params: Dict, camera_mats: Dict, save_data: Dict
) -> Dict:
    """Triangulate for multi-instance multi-channel.

    Args:
        n_cams (int): Numver of cameras
        sample_id (Text): Sample identifier.
        params (Dict): Parameters dictionary.
        camera_mats (Dict): Camera matrices dictioanry.
        save_data (Dict): Saved data dictionary.

    No Longer Returned:
        Dict: Updated saved data dictionary.
    """
    # Triangulate for all unique pairs
    save_data[sample_id]["triangulation"]["instances"] = []
    for instance in range(params["n_instances"]):
        for n_cam1 in range(n_cams):
            for n_cam2 in range(n_cam1 + 1, n_cams):
                pts1 = save_data[sample_id][params["camnames"][n_cam1]]["COM"][
                    instance, :
                ]
                pts2 = save_data[sample_id][params["camnames"][n_cam2]]["COM"][
                    instance, :
                ]
                pts1 = pts1[np.newaxis, :]
                pts2 = pts2[np.newaxis, :]

                test3d = ops.triangulate(
                    pts1,
                    pts2,
                    camera_mats[params["camnames"][n_cam1]],
                    camera_mats[params["camnames"][n_cam2]],
                ).squeeze()

                save_data[sample_id]["triangulation"][
                    "{}_{}".format(
                        params["camnames"][n_cam1], params["camnames"][n_cam2]
                    )
                ] = test3d

        pairs = [
            v for v in save_data[sample_id]["triangulation"].values() if len(v) == 3
        ]
        # import pdb
        # pdb.set_trace()
        pairs = np.stack(pairs, axis=1)
        final = np.nanmedian(pairs, axis=1).squeeze()
        save_data[sample_id]["triangulation"]["instances"].append(final[:, np.newaxis])
    return save_data


def triangulate_multi_instance_single_channel(
    n_cams: int,
    sample_id: Text,
    params: Dict,
    camera_mats: Dict,
    cameras: Dict,
    save_data: Dict,
) -> Dict:
    """Triangulate for multi-instance single-channel.

    Args:
        n_cams (int): Numver of cameras
        sample_id (Text): Sample identifier.
        params (Dict): Parameters dictionary.
        camera_mats (Dict): Camera matrices dictioanry.
        cameras (Dict): Camera dictionary.
        save_data (Dict): Saved data dictionary.

    No Longer Returned:
        Dict: Updated saved data dictionary.
    """
    # Go through the instances, adding the most parsimonious
    # points of the n_instances available points at each camera.
    cams = [camera_mats[params["camnames"][n_cam]] for n_cam in range(n_cams)]
    best_pts = []
    best_pts_inds = []
    for instance in range(params["n_instances"]):
        pts = []
        pts_inds = []

        # Build the initial list of points
        for n_cam in range(n_cams):
            pt = save_data[sample_id][params["camnames"][n_cam]]["COM"][instance, :]
            pt = pt[np.newaxis, :]
            pts.append(pt)
            pts_inds.append(instance)

        # Go through each camera (other than the first) and test
        # each instance
        for n_cam in range(1, n_cams):
            candidate_errors = []
            for n_point in range(params["n_instances"]):
                if len(best_pts_inds) >= 1:
                    if any(n_point == p[n_cam] for p in best_pts_inds):
                        candidate_errors.append(np.Inf)
                        continue

                pt = save_data[sample_id][params["camnames"][n_cam]]["COM"][n_point, :]
                pt = pt[np.newaxis, :]
                pts[n_cam] = pt
                pts_inds[n_cam] = n_point
                pts3d = ops.triangulate_multi_instance(pts, cams)

                # Loop through each camera, reproject the point
                # into image coordinates, and save the error.
                error = 0
                for n_proj in range(n_cams):
                    K = cameras[params["camnames"][n_proj]]["K"]
                    R = cameras[params["camnames"][n_proj]]["R"]
                    t = cameras[params["camnames"][n_proj]]["t"]
                    proj = ops.project_to2d(pts3d.T, K, R, t)
                    proj = proj[:, :2]
                    ref = save_data[sample_id][params["camnames"][n_proj]]["COM"][
                        pts_inds[n_proj], :
                    ]
                    error += np.sqrt(np.sum((proj - ref) ** 2))
                candidate_errors.append(error)

            # Keep the best instance combinations across cameras
            best_candidate = np.argmin(candidate_errors)
            pt = save_data[sample_id][params["camnames"][n_cam]]["COM"][
                best_candidate, :
            ]
            pt = pt[np.newaxis, :]
            pts[n_cam] = pt
            pts_inds[n_cam] = best_candidate

        best_pts.append(pts)
        best_pts_inds.append(pts_inds)

    # Do one final triangulation
    final3d = [
        ops.triangulate_multi_instance(best_pts[k], cams)
        for k in range(params["n_instances"])
    ]
    save_data[sample_id]["triangulation"]["instances"] = final3d
    return save_data