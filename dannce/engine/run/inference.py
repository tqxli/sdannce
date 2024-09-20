"""Handle inference for dannce and com networks.
"""
import os
import time
from typing import Dict, List, Text, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import dannce.engine.utils.image as image_utils
from dannce.config import print_and_set
from dannce.engine.utils.augmentation import construct_augmented_batch
from dannce.engine.utils.save import savedata_expval, savedata_tomat
from dannce.engine.utils.triangulation import (
    extract_multi_instance_multi_channel,
    extract_multi_instance_single_channel, extract_single_instance,
    triangulate_multi_instance_multi_channel,
    triangulate_multi_instance_single_channel, triangulate_single_instance)


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


def determine_com_save_filename(params: Dict):
    """Determine the filename for saving COM predictions.

    Args:
        params (Dict): Parameters dictionary.

    Returns:
        Text: Save filename.
    """
    if params["max_num_samples"] == "max":
        save_filename = "com3d"
    else:
        save_filename = "com3d%d" % (params["start_sample"])
    return save_filename


def infer_com_inference_range(
    params: Dict, predict_generator: torch.utils.data.Dataset,
):
    start_ind = params["start_sample"]
    end_ind = (
        np.min(
            [
                params["start_sample"] + params["max_num_samples"],
                len(predict_generator),
            ]
        )
        if params["max_num_samples"] != "max"
        else len(predict_generator)
    )
    return {"start_ind": start_ind, "end_ind": end_ind}


def infer_com(
    start_ind: int,
    end_ind: int,
    generator: torch.utils.data.Dataset,
    params: Dict,
    model: nn.Module,
    partition: Dict,
    save_data: Dict,
    camera_mats: Dict,
    cameras: Dict,
    device: torch.device,
    sample_save: int = 100,
):
    """Perform COM detection over a set of frames.
    
    Args:
        start_ind (int): Starting frame index
        end_ind (int): Ending frame index
        generator (torch.utils.data.Dataset): Data generator
        params (Dict): Parameters dictionary.
        model (Model): Inference model.
        partition (Dict): Partition dictionary
        save_data (Dict): Saved data dictionary
        camera_mats (Dict): Camera matrix dictionary
        cameras (Dict): Camera dictionary.
        sample_save (int, optional): Number of samples to use in fps estimation.
    """
    for n_frame in tqdm(range(start_ind, end_ind)):
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


def infer_dannce_inference_range(
    params: Dict, generator: torch.utils.data.Dataset,
):
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

    start_ind = int(params["start_batch"])
    end_ind = int(params["maxbatch"])
    return start_ind, end_ind, bs


def infer_dannce(
    generator: torch.utils.data.Dataset,
    params: Dict,
    model: nn.Module,
    partition: Dict,
    device: Text,
    n_chn: int,
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
    start_ind, end_ind, bs = infer_dannce_inference_range(params, generator)
    save_data = {}

    if save_heatmaps:
        save_path = os.path.join(params["dannce_predict_dir"], "heatmaps")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    pbar = tqdm(range(start_ind, end_ind))
    for idx, i in enumerate(pbar):

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
                (xcoord, ycoord, zcoord,) = image_utils.plot_markers_3d_torch(preds)
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
                    im = image_utils.norm_im(preds[..., i].cpu().numpy()) * 255
                    im = im.astype("uint8")
                    of = os.path.join(savedir, f"{sampleID}_{i}.tif")
                    imageio.mimwrite(of, np.transpose(im, [2, 0, 1]))

    return save_data


def save_inference_checkpoint(
    params: Dict, save_data: Dict, num_markers: int, savename: str,
):
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
    generator: torch.utils.data.Dataset,
    params: Dict,
    custom_model_params: Dict,
    model: nn.Module,
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
            volumes_train, grid_centers_train = construct_augmented_batch(
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
