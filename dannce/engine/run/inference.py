"""Handle inference procedures for dannce and com networks.
"""
import numpy as np
import os
import time
import dannce.engine.data.processing as processing
from dannce.engine.data import ops
from dannce.config import print_and_set
from typing import List, Dict, Text, Tuple, Union
import torch
import matplotlib
from dannce.engine.data.processing import savedata_tomat, savedata_expval

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm


def print_checkpoint(
    n_frame: int, start_ind: int, end_time: float, sample_save: int = 100
) -> float:
    """Print checkpoint messages indicating frame and fps for inference.

    Args:
        n_frame (int): Frame number
        start_ind (int): Start index
        end_time (float): Timing reference
        sample_save (int, optional): Number of samples to use in fps estimation.

    No Longer Returned:
        float: New timing reference.
    """
    print("Predicting on sample %d" % (n_frame), flush=True)
    if (n_frame - start_ind) % sample_save == 0 and n_frame != start_ind:
        print(n_frame)
        print("{} samples took {} seconds".format(sample_save, time.time() - end_time))
        end_time = time.time()
    return end_time


def predict_batch(model, generator, n_frame: int, params: Dict, device) -> np.ndarray:
    """Predict for a single batch and reformat output.

    Args:
        model (Model): interence model
        generator (keras.utils.Sequence): Data generator
        n_frame (int): Frame number
        params (Dict): Parameters dictionary.

    No Longer Returned:
        np.ndarray: n_batch x n_cam x h x w x c predictions
    """
    im = (
        torch.from_numpy(generator.__getitem__(n_frame)[0])
        .permute(0, 3, 1, 2)
        .to(device)
    )
    pred = model(im)
    if params["mirror"]:
        n_cams = 1
    else:
        n_cams = len(params["camnames"])
    shape = [-1, n_cams, *pred.shape[1:]]
    pred = pred.reshape(*shape).permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    return pred


def debug_com(
    params: Dict,
    pred: np.ndarray,
    pred_batch: np.ndarray,
    generator,
    ind: np.ndarray,
    n_frame: int,
    n_batch: int,
    n_cam: int,
    n_instance=None,
):
    """Print useful figures for COM debugging.

    Args:
        params (Dict): Parameters dictionary.
        pred (np.ndarray): Reformatted batch predictions.
        pred_batch (np.ndarray): Batch prediction.
        generator (keras.utils.Sequence): DataGenerator
        ind (np.ndarray): Prediction in image indices
        n_frame (int): Frame number
        n_batch (int): Batch number
        n_cam (int): Camera number
    """
    com_predict_dir = params["com_predict_dir"]
    cmapdir = os.path.join(com_predict_dir, "cmap")
    overlaydir = os.path.join(com_predict_dir, "overlay")
    if not os.path.exists(cmapdir):
        os.makedirs(cmapdir)
    if not os.path.exists(overlaydir):
        os.makedirs(overlaydir)
    print("Writing " + params["com_debug"] + " confidence maps to " + cmapdir)
    print("Writing " + params["com_debug"] + "COM-image overlays to " + overlaydir)

    batch_size = pred_batch.shape[0]
    # Write preds
    plt.figure(0)
    plt.cla()
    pred_to_plot = np.squeeze(pred[n_cam])
    fname = os.path.join(
        cmapdir, params["com_debug"] + str(n_frame * batch_size + n_batch) + ".png",
    )
    if n_instance is not None:
        pred_to_plot = pred_to_plot[..., n_instance]
        fname = fname.replace(".png", "_0{}.png".format(n_instance))
    plt.imshow(pred_to_plot)
    plt.savefig(fname)

    plt.figure(1)
    plt.cla()
    im = generator.__getitem__(n_frame * batch_size + n_batch)
    plt.imshow(processing.norm_im(im[0][n_cam]))
    plt.plot(
        (ind[0] - params["crop_width"][0]) / params["downfac"],
        (ind[1] - params["crop_height"][0]) / params["downfac"],
        "or",
    )
    fname = fname.replace(cmapdir, overlaydir)
    plt.savefig(fname)


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
            processing.get_peak_inds_multi_instance(
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
            np.array(processing.get_peak_inds(np.squeeze(pred[n_cam, :, :, instance])))
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
        np.array(processing.get_peak_inds(np.squeeze(pred[n_cam]))) * params["downfac"]
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


def infer_com(
    start_ind: int,
    end_ind: int,
    generator,
    params: Dict,
    model,
    partition: Dict,
    save_data: Dict,
    camera_mats: Dict,
    cameras: Dict,
    device,
    sample_save: int = 100,
):
    """Perform COM detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        camera_mats (Dict): Camera matrix dictionary
        cameras (Dict): Camera dictionary.
        sample_save (int, optional): Number of samples to use in fps estimation.
    """
    end_time = time.time()
    for n_frame in tqdm(range(start_ind, end_ind)):
        # end_time = print_checkpoint(
        #    n_frame, start_ind, end_time, sample_save=sample_save
        # )
        pred_batch = predict_batch(model, generator, n_frame, params, device)
        n_batches = pred_batch.shape[0]

        for n_batch in range(n_batches):
            # By selecting -1 for the last axis, we get the COM index for a
            # normal COM network, and also the COM index for a multi_mode COM network,
            # as in multimode the COM label is put at the end
            if params["mirror"] and params["n_instances"] == 1:
                # For mirror we need to reshape pred so that the cameras are in front, so
                # it works with the downstream code
                pred = pred_batch[n_batch, 0]
                pred = np.transpose(pred, (2, 0, 1))
            elif params["mirror"]:
                raise Exception(
                    "mirror mode with multiple animal instances not currently supported."
                )
            elif params["n_instances"] > 1 and params["n_channels_out"] > 1:
                pred = pred_batch[n_batch, ...]
            else:
                pred = pred_batch[n_batch, :, :, :, -1]
            sample_id = partition["valid_sampleIDs"][n_frame * n_batches + n_batch]
            save_data[sample_id] = {}
            save_data[sample_id]["triangulation"] = {}
            n_cams = pred.shape[0]

            for n_cam in range(n_cams):
                args = [
                    pred,
                    pred_batch,
                    n_cam,
                    sample_id,
                    n_frame,
                    n_batch,
                    params,
                    save_data,
                    cameras,
                    generator,
                ]
                if params["n_instances"] == 1:
                    save_data = extract_single_instance(*args)
                elif params["n_channels_out"] == 1:
                    save_data = extract_multi_instance_single_channel(*args)
                elif params["n_channels_out"] > 1:
                    save_data = extract_multi_instance_multi_channel(*args)

            # Handle triangulation for single or multi instance
            if params["n_instances"] == 1:
                save_data = triangulate_single_instance(
                    n_cams, sample_id, params, camera_mats, save_data
                )
            elif params["n_channels_out"] == 1:
                save_data = triangulate_multi_instance_single_channel(
                    n_cams, sample_id, params, camera_mats, cameras, save_data
                )
            elif params["n_channels_out"] > 1:
                save_data = triangulate_multi_instance_multi_channel(
                    n_cams, sample_id, params, camera_mats, save_data
                )
    return save_data


def infer_dannce(
    generator,
    params: Dict,
    model,
    partition: Dict,
    device: Text,
    n_chn: int,
    sil_generator=None,
    save_heatmaps=False,
):
    """Perform dannce detection over a set of frames.

    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (keras.utils.Sequence): Keras data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        device (Text): Gpu device name
        n_chn (int): Number of output channels
    """
    n_frames = len(generator)
    bs = params["batch_size"]
    generator_maxbatch = np.ceil(n_frames / bs)

    if params["maxbatch"] != "max" and params["maxbatch"] > generator_maxbatch:
        print(
            "Maxbatch was set to a larger number of matches than exist in the video. Truncating."
        )
        print_and_set(params, "maxbatch", generator_maxbatch)

    if params["maxbatch"] == "max":
        print_and_set(params, "maxbatch", generator_maxbatch)

    end_time = time.time()
    save_data = {}
    start_ind = int(params["start_batch"])
    end_ind = int(params["maxbatch"])

    if save_heatmaps:
        save_path = os.path.join(params["dannce_predict_dir"], "heatmaps")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    pbar = tqdm(range(start_ind, end_ind))
    for idx, i in enumerate(pbar):
        # print("Predicting on batch {}".format(i), flush=True)
        # if (i - start_ind) % 10 == 0 and i != start_ind:
        # print(i)
        # print("10 batches took {} seconds".format(time.time() - end_time))
        # end_time = time.time()

        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print("Saving checkpoint at {}th batch".format(i))
            if params["expval"]:
                p_n = savedata_expval(
                    params["dannce_predict_dir"] + "/save_data_AVG.mat",
                    params,
                    write=True,
                    data=save_data,
                    tcoord=False,
                    num_markers=n_chn,
                    pmax=True,
                )
            else:
                p_n = savedata_tomat(
                    params["dannce_predict_dir"] + "/save_data_MAX.mat",
                    params,
                    params["vmin"],
                    params["vmax"],
                    params["nvox"],
                    write=True,
                    data=save_data,
                    num_markers=n_chn,
                    tcoord=False,
                )

        ims = generator.__getitem__(i)

        vols = torch.from_numpy(ims[0][0]).permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
        # replace occluded view
        if params["downscale_occluded_view"]:
            occlusion_scores = ims[0][2]
            occluded_views = occlusion_scores > 0.5

            vols = vols.reshape(
                vols.shape[0], -1, 3, *vols.shape[2:]
            )  # [B, 6, 3, H, W, D]

            for instance in range(occluded_views.shape[0]):
                occluded = np.where(occluded_views[instance])[0]
                unoccluded = np.where(~occluded_views[instance])[0]
                for view in occluded:
                    alternative = np.random.choice(unoccluded)
                    vols[instance][view] = vols[instance][alternative]
                    print(f"Replace view {view} with {alternative}")

            vols = vols.reshape(vols.shape[0], -1, *vols.shape[3:])

        model_inputs = [vols.to(device)]
        if params["expval"]:
            model_inputs.append(torch.from_numpy(ims[0][1]).to(device))
        else:
            model_inputs.append(None)

        with torch.no_grad():
            pred = model(*model_inputs)

        if params["expval"]:
            probmap = (
                torch.amax(pred[1], dim=(2, 3, 4)).squeeze(0).detach().cpu().numpy()
            )
            heatmaps = pred[1].squeeze().detach().cpu().numpy()
            pred = pred[0].detach().cpu().numpy()
            for j in range(pred.shape[0]):
                pred_max = probmap[j]
                sampleID = partition["valid_sampleIDs"][i * bs + j]
                save_data[idx * bs + j] = {
                    "pred_max": pred_max,
                    "pred_coord": pred[j],
                    "sampleID": sampleID,
                }
                if save_heatmaps:
                    np.save(os.path.join(save_path, sampleID), heatmaps[j])

        else:
            pred = pred[1]
            for j in range(pred.shape[0]):
                preds = pred[j].permute(1, 2, 3, 0).detach()
                pred_max = preds.max(0).values.max(0).values.max(0).values
                pred_total = preds.sum((0, 1, 2))
                (xcoord, ycoord, zcoord,) = processing.plot_markers_3d_torch(preds)
                coord = torch.stack([xcoord, ycoord, zcoord])
                pred_log = pred_max.log() - pred_total.log()
                sampleID = partition["valid_sampleIDs"][i * bs + j]

                save_data[idx * bs + j] = {
                    "pred_max": pred_max.cpu().numpy(),
                    "pred_coord": coord.cpu().numpy(),
                    "true_coord_nogrid": ims[1][0][j],
                    "logmax": pred_log.cpu().numpy(),
                    "sampleID": sampleID,
                }

                # save predicted heatmaps
                savedir = "./debug_MAX"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                for i in range(preds.shape[-1]):
                    im = processing.norm_im(preds[..., i].cpu().numpy()) * 255
                    im = im.astype("uint8")
                    of = os.path.join(savedir, f"{sampleID}_{i}.tif")
                    imageio.mimwrite(of, np.transpose(im, [2, 0, 1]))

    return save_data


def save_inference_checkpoint(params, save_data, num_markers, savename):
    p_n = savedata_expval(
        params["dannce_predict_dir"] + "/{}".format(savename),
        params,
        write=True,
        data=save_data,
        tcoord=False,
        num_instances=params["n_instances"],
        num_markers=num_markers,
        pmax=True,
    )


def infer_sdannce(
    generator,
    params: Dict,
    custom_model_params: Dict,
    model,
    partition: Dict,
    device: Text,
):
    n_frames = len(generator)
    bs = params["batch_size"]
    generator_maxbatch = int(np.ceil(n_frames / bs))

    if params["maxbatch"] != "max" and params["maxbatch"] > generator_maxbatch:
        print(
            "Maxbatch was set to a larger number of matches than exist in the video. Truncating."
        )
        print_and_set(params, "maxbatch", generator_maxbatch)

    if params["maxbatch"] == "max":
        print_and_set(params, "maxbatch", generator_maxbatch)

    save_data, save_data_init = {}, {}
    start_ind = params["start_batch"]
    end_ind = params["maxbatch"]
    max_num_sample = (
        params["max_num_samples"] if params["maxbatch"] != "max" else n_frames
    )

    pbar = tqdm(range(start_ind, end_ind))
    for idx, i in enumerate(pbar):

        if (i - start_ind) % 1000 == 0 and i != start_ind:
            print("Saving checkpoint at {}th batch".format(i))
            save_inference_checkpoint(
                params, save_data, final_poses.shape[-1], "/save_data_AVG.mat"
            )
            save_inference_checkpoint(
                params, save_data, init_poses.shape[-1], "/init_save_data_AVG.mat"
            )

        # retrieve batched inputs
        indices = np.arange(i * bs, min((i + 1) * bs, max_num_sample, n_frames))
        ims = generator.get_batched(indices)
        vols = torch.from_numpy(ims[0]).permute(0, 4, 1, 2, 3)

        # replace occluded view
        if params["downscale_occluded_view"]:
            occlusion_scores = ims[0][2]
            occluded_views = occlusion_scores > 0.5

            vols = vols.reshape(
                vols.shape[0], -1, 3, *vols.shape[2:]
            )  # [B, 6, 3, H, W, D]

            for instance in range(occluded_views.shape[0]):
                occluded = np.where(occluded_views[instance])[0]
                unoccluded = np.where(~occluded_views[instance])[0]
                for view in occluded:
                    alternative = np.random.choice(unoccluded)
                    vols[instance][view] = vols[instance][alternative]
                    print(f"Replace view {view} with {alternative}")

            vols = vols.reshape(vols.shape[0], -1, *vols.shape[3:])

        if (
            isinstance(params["replace_view"], int)
            and params["replace_view"] <= params["n_views"]
        ):
            vols = vols.reshape(
                vols.shape[0], -1, 3, *vols.shape[2:]
            )  # [B, 6, 3, H, W, D]
            for batch in range(vols.shape[0]):
                alternative_view = np.random.choice(
                    np.delete(np.arange(params["n_views"]), params["replace_view"] - 1)
                )
                assert alternative_view != params["replace_view"] - 1
                vols[batch, params["replace_view"] - 1] = vols[batch, alternative_view]
            vols = vols.reshape(vols.shape[0], -1, *vols.shape[3:])

        model_inputs = [vols.to(device)]
        grid_centers = torch.from_numpy(ims[1]).to(device)
        model_inputs.append(grid_centers)

        try:
            init_poses, heatmaps, inter_features = model.pose_generator(*model_inputs)
            final_poses = model.inference(
                init_poses, grid_centers, heatmaps, inter_features
            )

            if custom_model_params.get("relpose", True):
                com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1)  # [N, 3, 1]
                nvox = round(grid_centers.shape[1] ** (1 / 3))
                vsize = (
                    grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()
                ) / nvox
                final_poses = final_poses * vsize
                if not custom_model_params.get("predict_diff", True):
                    final_poses += com3d

            if custom_model_params.get("predict_diff", True):
                final_poses += init_poses[..., : final_poses.shape[-1]]
        except:
            final_poses, heatmaps, _ = model(*model_inputs)
            init_poses = final_poses
            # params["n_instances"] = 2

        probmap = torch.amax(heatmaps, dim=(2, 3, 4)).squeeze(0).detach().cpu()
        probmap = probmap.reshape(-1, params["n_instances"], *probmap.shape[1:]).numpy()
        heatmaps = (
            heatmaps.squeeze()
            .detach()
            .cpu()
            .reshape(-1, params["n_instances"], *heatmaps.shape[1:])
            .numpy()
        )
        pred = (
            final_poses.detach()
            .cpu()
            .reshape(-1, params["n_instances"], *final_poses.shape[1:])
            .numpy()
        )
        pred_init = (
            init_poses.detach()
            .cpu()
            .reshape(-1, params["n_instances"], *init_poses.shape[1:])
            .numpy()
        )

        for j in range(pred.shape[0]):
            pred_max = probmap[j]
            sampleID = partition["valid_sampleIDs"][i * bs + j]
            save_data[idx * bs + j] = {
                "pred_max": pred_max,
                "pred_coord": pred[j],
                "sampleID": sampleID,
            }
            save_data_init[idx * bs + j] = {
                "pred_max": pred_max,
                "pred_coord": pred_init[j],
                "sampleID": sampleID,
            }

    if params["save_tag"] is not None:
        savename = "save_data_AVG%d.mat" % (params["save_tag"])
    else:
        savename = "save_data_AVG.mat"
    save_inference_checkpoint(params, save_data, final_poses.shape[-1], savename)
    save_inference_checkpoint(
        params, save_data_init, init_poses.shape[-1], "init_save_data_AVG.mat"
    )


def save_results(params, save_data):
    if params["expval"]:
        if params["save_tag"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"],
                "save_data_AVG%d.mat" % (params["save_tag"]),
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_AVG.mat")
        p_n = savedata_expval(
            path,
            params,
            write=True,
            data=save_data,
            tcoord=False,
            num_markers=params["n_markers"],
            pmax=True,
        )
    else:
        if params["save_tag"] is not None:
            path = os.path.join(
                params["dannce_predict_dir"],
                "save_data_MAX%d.mat" % (params["save_tag"]),
            )
        else:
            path = os.path.join(params["dannce_predict_dir"], "save_data_MAX.mat")
        p_n = savedata_tomat(
            path,
            params,
            params["vmin"],
            params["vmax"],
            params["nvox"],
            write=True,
            data=save_data,
            num_markers=params["n_markers"],
            tcoord=False,
        )


def inference_ttt(
    generator,
    params,
    model,
    optimizer,
    device,
    partition,
    online=False,
    niter=10,
    gcn=False,
    transform=False,
    downsample=1,
):
    from dannce.engine.trainer.train_utils import LossHelper

    ckpt = torch.load(params["dannce_predict_model"])
    if online:
        model.load_state_dict(ckpt["state_dict"])

    # freeze stages
    if gcn:
        for name, param in model.named_parameters():
            if ("pose_generator" in name) and ("output" not in name):
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if "output" not in name:
                param.requires_grad = False

    criterion = LossHelper(params)

    if params["maxbatch"] != "max" and params["maxbatch"] > len(generator):
        print(
            "Maxbatch was set to a larger number of matches than exist in the video. Truncating"
        )
        print_and_set(params, "maxbatch", len(generator))

    if params["maxbatch"] == "max":
        print_and_set(params, "maxbatch", len(generator))

    save_data = {}
    start_ind = params["start_batch"]
    end_ind = params["maxbatch"]

    pbar = tqdm(range(start_ind, end_ind, downsample))
    for idx, i in enumerate(pbar):
        if not online:
            model.load_state_dict(ckpt["state_dict"])

        batch = generator.__getitem__(i)
        batch = [*batch[0], *batch[1]]
        volumes = torch.from_numpy(batch[0]).float().permute(0, 4, 1, 2, 3).to(device)
        grid_centers = torch.from_numpy(batch[1]).float().to(device)
        keypoints_3d_gt = torch.from_numpy(batch[2]).float().to(device)
        aux = None
        inputs = [volumes, grid_centers]

        # form each batch with transformed versions of a single test data
        if transform:
            volumes_train, grid_centers_train = form_batch(
                volumes.permute(0, 2, 3, 4, 1), grid_centers
            )
            volumes_train = volumes_train.permute(0, 4, 1, 2, 3)
            inputs = [volumes_train, grid_centers_train]

        model.train()
        for _ in range(niter):
            optimizer.zero_grad()

            if gcn:
                init_poses, heatmaps, inter_features = model.pose_generator(*inputs)
                final_poses = model.inference(
                    init_poses, grid_centers, heatmaps, inter_features
                )
                nvox = round(grid_centers.shape[1] ** (1 / 3))
                vsize = (
                    grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()
                ) / nvox
                final_poses = final_poses * vsize
                final_poses += init_poses
                keypoints_3d_pred = final_poses
            else:
                keypoints_3d_pred, heatmaps, _ = model(*inputs)

            bone_loss = criterion.compute_loss(
                keypoints_3d_gt, keypoints_3d_pred, heatmaps, grid_centers, aux
            )[0]

            if transform:
                consist_loss = torch.abs(torch.diff(keypoints_3d_pred, dim=0)).mean()
            else:
                consist_loss = bone_loss.new_zeros(())

            total_loss = bone_loss + consist_loss

            pbar.set_description(
                "Consistency Loss {:.4f} Bone Loss {:.4f}".format(
                    consist_loss.item(), bone_loss.item()
                )
            )

            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if gcn:
                init_poses, heatmaps, inter_features = model.pose_generator(
                    volumes, grid_centers
                )
                final_poses = model.inference(
                    init_poses, grid_centers, heatmaps, inter_features
                )
                nvox = round(grid_centers.shape[1] ** (1 / 3))
                vsize = (
                    grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()
                ) / nvox
                final_poses = final_poses * vsize
                final_poses += init_poses
                pred = [final_poses, heatmaps]
            else:
                pred = model(volumes, grid_centers)

            probmap = (
                torch.amax(pred[1], dim=(2, 3, 4)).squeeze(0).detach().cpu().numpy()
            )
            heatmaps = pred[1].squeeze().detach().cpu().numpy()
            pred = pred[0].detach().cpu().numpy()
            for j in range(pred.shape[0]):
                pred_max = probmap[j]
                sampleID = partition["valid_sampleIDs"][i * pred.shape[0] + j]
                save_data[idx * pred.shape[0] + j] = {
                    "pred_max": pred_max,
                    "pred_coord": pred[j],
                    "sampleID": sampleID,
                }

    if online:
        state = {
            "state_dict": model.state_dict(),
            "params": ckpt["params"],
            "optimizer": ckpt["optimizer"],
            "epoch": ckpt["epoch"],
        }
        torch.save(
            state,
            os.path.join(
                params["dannce_predict_dir"],
                "checkpoint-online-iter{}.pth".format(
                    ((end_ind - start_ind) // downsample) * niter
                ),
            ),
        )

    return save_data


def random_rotate(X, y_3d, aux=None):
    def rot90(X):
        X = X.permute(1, 0, 2, 3)
        return X.flip(1)

    def rot180(X):
        return X.flip(0).flip(1)

    rot = np.random.choice(np.arange(4), 1)
    for j in range(X.shape[0]):
        if rot == 0:
            pass
        elif rot == 1:
            # Rotate180
            X[j], y_3d[j] = rot180(X[j]), rot180(y_3d[j])
            if aux is not None:
                aux[j] = rot180(aux[j])
        elif rot == 2:
            # Rotate90
            X[j], y_3d[j] = rot90(X[j]), rot90(y_3d[j])
            if aux is not None:
                aux[j] = rot90(aux[j])
        elif rot == 3:
            # Rotate -90/270
            X[j], y_3d[j] = rot180(rot90(X[j])), rot180(rot90(y_3d[j]))
            if aux is not None:
                aux[j] = rot180(rot90(aux[j]))
    return X, y_3d, aux


def apply_random_transforms(volumes, grids, aux=None):
    grids = grids.reshape(grids.shape[0], 80, 80, 80, 3)

    volumes, grids, aux = random_rotate(volumes, grids, aux)
    grids = grids.reshape(grids.shape[0], -1, 3)
    return volumes, grids, aux


def form_batch(volumes, grids, aux=None, copies_per_sample=1):
    copies = []
    n_samples = volumes.shape[0]
    for i in range(n_samples):
        copies += [
            apply_random_transforms(
                volumes[i].clone().unsqueeze(0),
                grids[i].clone().unsqueeze(0),
                aux[i].clone().unsqueeze(0) if aux is not None else aux,
            )
            for _ in range(copies_per_sample)
        ]

    volumes = torch.cat([copy[0] for copy in copies], dim=0)
    grids = torch.cat([copy[1] for copy in copies], dim=0)
    aux = torch.cat([copy[2] for copy in copies], dim=0) if aux is not None else aux

    return volumes, grids, aux
