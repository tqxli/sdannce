import os
import numpy as np
import scipy.io as sio
import imageio
import tqdm
import argparse
from projection import *
from dannce.engine.skeletons.utils import load_body_profile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def main(args):
    EXP_ROOT = args.root
    EXP = args.pred
    DATA_FILE = 'save_data_AVG0.mat'
    LABEL3D_FILE = [f for f in os.listdir(EXP_ROOT) if f.endswith('dannce.mat')][0]
    N_FRAMES = args.n_frames
    VID_NAME = "0.mp4"
    START_FRAME = args.start_frame
    CAMERAS = ["Camera{}".format(int(i)) for i in args.cameras.split(',')]
    ANIMAL= args.skeleton
    COLOR = ['white', 'yellow']
    CONNECTIVITY = load_body_profile(ANIMAL)["limbs"]

    vid_path = os.path.join(EXP_ROOT, 'videos') 
    SAVE_ROOT = os.path.join(EXP_ROOT, EXP)
    save_path = os.path.join(SAVE_ROOT, 'vis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    camnames = "Camera{}".format(args.cameras)
    fname = f'frame{START_FRAME}-{START_FRAME+N_FRAMES}_{camnames}.mp4'

    ###############################################################################################################
    # load camera parameters
    cameras = load_cameras(os.path.join(EXP_ROOT, LABEL3D_FILE))

    pred_path = os.path.join(EXP_ROOT, EXP)
    pose_3d = sio.loadmat(os.path.join(pred_path, DATA_FILE))['pred'][START_FRAME: START_FRAME+N_FRAMES]
    pred_3d1 = pose_3d[:, 0, :, :]
    pred_3d2 = pose_3d[:, 1, :, :]
    com_3d = sio.loadmat(os.path.join(pred_path, 'com3d_used.mat'))['com'][START_FRAME: START_FRAME+N_FRAMES]
    com_3d1 = com_3d[:, :, 0]
    com_3d2 = com_3d[:, :, 1]  

    # compute projections
    pred_2d_1, pred_2d_2, com_2d_1, com_2d_2 = {}, {}, {}, {}
    pose_3d1 = pred_3d1
    if pose_3d1.shape[-1] != 3:
        pose_3d1 = np.transpose(pred_3d1, (0, 2, 1))
    com_3d1 = np.expand_dims(com_3d1, 1)

    pose_3d2 = pred_3d2
    if pose_3d2.shape[-1] != 3:
        pose_3d2 = np.transpose(pose_3d2, (0, 2, 1))
    com_3d2 = np.expand_dims(com_3d2, 1)

    n_joints = pose_3d1.shape[1]
    pts = np.concatenate((pose_3d1, pose_3d2, com_3d1, com_3d2), axis=1)
    num_chan = pts.shape[1]
    pts = np.reshape(pts, (-1, 3))
    for cam in CAMERAS:
        projpts = project_to_2d(pts,
                                cameras[cam]["K"],
                                cameras[cam]["r"],
                                cameras[cam]["t"])[:, :2]

        projpts = distortPoints(projpts,
                                cameras[cam]["K"],
                                np.squeeze(cameras[cam]["RDistort"]),
                                np.squeeze(cameras[cam]["TDistort"]))
        projpts = projpts.T
        projpts = np.reshape(projpts, (-1, num_chan, 2))

        pred_2d_1[cam] = projpts[:, :n_joints, :]
        pred_2d_2[cam] = projpts[:, n_joints:2*n_joints, :]
        com_2d_1[cam] = projpts[:, -2:-1, :]
        com_2d_2[cam] = projpts[:, -1:, :]
    del projpts, pred_3d1, com_3d1, com_3d2

    # open videos
    vids = [imageio.get_reader(os.path.join(vid_path, cam, VID_NAME)) for cam in CAMERAS]

    # set up video writer
    metadata = dict(title='SDANNCE', artist='Matplotlib')
    writer = FFMpegWriter(fps=50, metadata=metadata)

    ###############################################################################################################
    # setup figure
    n_cams = len(CAMERAS)
    n_rows = int(np.ceil(n_cams / 3))
    n_cols = n_cams % 3 if n_cams < 3 else 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4))
    if len(CAMERAS) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    frames_to_plot = np.arange(START_FRAME, N_FRAMES+START_FRAME) 

    with writer.saving(fig, os.path.join(save_path, fname), dpi=300):
        for idx, curr_frame in enumerate(tqdm.tqdm(frames_to_plot)):
            # grab images
            imgs = [vid.get_data(curr_frame) for vid in vids]

            for i, cam in enumerate(CAMERAS):
                rat1_kpts_2d = pred_2d_1[cam][idx]
                rat1_com = com_2d_1[cam][idx]

                axes[i].imshow(imgs[i])
                axes[i].scatter(rat1_com[:, 0], rat1_com[:, 1], marker='.', color='blue', linewidths=1)

                for (index_from, index_to) in CONNECTIVITY:
                    xs, ys = [np.array([rat1_kpts_2d[index_from, j], rat1_kpts_2d[index_to, j]]) for j in range(2)]
                    axes[i].plot(xs, ys, c=COLOR[1], lw=1, alpha=0.8, markersize=2, markerfacecolor='blue', marker='o', markeredgewidth=0.4)
                    del xs, ys
         
                rat2_kpts_2d = pred_2d_2[cam][idx]            
                rat2_com = com_2d_2[cam][idx]
                axes[i].scatter(rat2_com[:, 0], rat2_com[:, 1], marker='.', color='red', linewidths=1)

                for (index_from, index_to) in CONNECTIVITY:
                    xs, ys = [np.array([rat2_kpts_2d[index_from, j], rat2_kpts_2d[index_to, j]]) for j in range(2)]
                    axes[i].plot(xs, ys, c=COLOR[0], lw=1, alpha=0.8, markersize=2, markerfacecolor='red', marker='o', markeredgewidth=0.4)
                    del xs, ys
                
                axes[i].set_title(CAMERAS[i])
                axes[i].axis("off")
                
            fig.suptitle("Frame: {}".format(curr_frame))
            fig.tight_layout()
            
            writer.grab_frame()
            for i in range(len(CAMERAS)):
                axes[i].clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--n_instances", type=int, default=2, help="number of animals present in the scene (1 or 2)")
    parser.add_argument("--skeleton", type=str, default="rat23", help="corresponding skeleton connectivity for the animal, see ./skeletons")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--n_frames", type=int, help="number of frames to plot", default=10)
    parser.add_argument("--fps", default=50, type=int)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--cameras", type=str, default="1", help="camera(s) to plot")

    args = parser.parse_args()
    main(args)
