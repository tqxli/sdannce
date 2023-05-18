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
    DATA_FILE = args.datafile
    datafile_start = int(DATA_FILE.split('save_data_AVG')[-1].split('.mat')[0])
    try:
        datafile_start = int(datafile_start)
    except ValueError:
        raise Exception('Datafile not named as save_data_AVG%n.mat')
    LABEL3D_FILE = [f for f in os.listdir(EXP_ROOT) if f.endswith('dannce.mat')][0]
    N_FRAMES = args.n_frames
    VID_NAME = "0.mp4"
    START_FRAME = args.start_frame
    CAMERAS = ["Camera{}".format(int(i)) for i in args.cameras.split(',')]
    ANIMAL= args.skeleton
    N_ANIMALS = args.n_animals
    MARKER_COLOR = ['blue', 'red']
    COLOR = ['yellow', 'white']
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
    # prediction file might not start at frame 0
    pose_3d = sio.loadmat(os.path.join(pred_path, DATA_FILE))['pred'][
        START_FRAME-datafile_start: START_FRAME-datafile_start+N_FRAMES
    ]

    # pose_3d: [N, n_instances, n_landmarks, 3]
    assert pose_3d.shape[1] == N_ANIMALS, "The specified number of animals does not match predictions"
    if pose_3d.shape[-1] != 3:
        pose_3d = pose_3d.transpose((0, 1, 3, 2))
    n_kpts = pose_3d.shape[-2]

    # load centers of masses used for predictions
    # com_3d: [N, 3, n+instances]
    com_3d = sio.loadmat(os.path.join(pred_path, 'com3d_used.mat'))['com'][START_FRAME: START_FRAME+N_FRAMES]
    if N_ANIMALS == 1:
        assert len(com_3d.shape) == 2 and com_3d.shape[-1] == 3
        com_3d = com_3d[:, None, None, :]
    else:
        assert com_3d.shape[-1] == N_ANIMALS
        com_3d = com_3d.transpose((0, 2, 1)) # [N, N_ANIMALS, 3]
        com_3d = np.expand_dims(com_3d, axis=2) #[N, 1, 3]

    pts = np.concatenate((
        pose_3d.reshape((pose_3d.shape[0], -1, *pose_3d.shape[3:])),
        com_3d.reshape((com_3d.shape[0], -1, *com_3d.shape[3:]))
    ), axis=1)
    num_chan = pts.shape[1]
    pts = pts.reshape((-1, 3))
    
    pred_2d, com_2d = {}, {}
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

        proj_kpts = projpts[:, :N_ANIMALS*n_kpts]
        pred_2d[cam] = proj_kpts.reshape((proj_kpts.shape[0], N_ANIMALS, n_kpts, 2))
        proj_com = projpts[:, N_ANIMALS*n_kpts:] 
        com_2d[cam] = proj_com.reshape(proj_com.shape[0], N_ANIMALS, 1, 2)

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
                axes[i].imshow(imgs[i])
                
                kpts_2d = pred_2d[cam][idx]
                com = com_2d[cam][idx]

                for ani in range(N_ANIMALS):
                    axes[i].scatter(com[ani, :, 0], com[ani, :, 1], marker='.', color=MARKER_COLOR[ani], linewidths=1)
                    for (index_from, index_to) in CONNECTIVITY:
                        xs, ys = [np.array([kpts_2d[ani, index_from, j], kpts_2d[ani, index_to, j]]) for j in range(2)]
                        axes[i].plot(xs, ys, c=COLOR[ani], lw=1, alpha=0.8, markersize=2, markerfacecolor=MARKER_COLOR[ani], marker='o', markeredgewidth=0.4)
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
    parser.add_argument("--datafile", type=str, default='save_data_AVG0.mat', help='name of the saved prediction file')
    parser.add_argument("--skeleton", type=str, default="rat23", help="corresponding skeleton connectivity for the animal, see ./skeletons")
    parser.add_argument("--n_animals", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--n_frames", type=int, help="number of frames to plot", default=10)
    parser.add_argument("--fps", default=50, type=int)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument("--cameras", type=str, default="1", help="camera(s) to plot")

    args = parser.parse_args()
    main(args)
