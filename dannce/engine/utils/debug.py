import os
from typing import Dict, List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

import dannce.engine.utils.image as image_utils


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
    plt.imshow(image_utils.norm_im(im[0][n_cam]))
    plt.plot(
        (ind[0] - params["crop_width"][0]) / params["downfac"],
        (ind[1] - params["crop_height"][0]) / params["downfac"],
        "or",
    )
    fname = fname.replace(cmapdir, overlaydir)
    plt.savefig(fname)


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
        image_utils.plot_markers_2d(image_utils.norm_im(imo), lo, newfig=False)
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
    params: Dict, tifdir: str, X: np.ndarray, sampleIDs: List, n_cams: int,
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
            im = image_utils.norm_im(im) * 255
            im = im.astype("uint8")
            of = os.path.join(tifdir, str(sampleIDs[i]) + "_cam" + str(j) + ".tif",)
            imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))


def generate_visual_hull(
    aux: np.ndarray, sampleIDs: List, savedir: str = "./visual_hull",
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