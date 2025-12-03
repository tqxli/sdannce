import os
import numpy as np
import yaml
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt

from dannce.engine.data.io import load_camera_params, load_labels, load_camnames
from dannce.engine.utils.projection import distortPoints, project_to_2d
from dannce.engine.skeletons.utils import load_body_profile

IO_FILE = "/mnt/data0/motion_exps/Mouse61482_113023/io.yaml"
dataroot = os.path.dirname(IO_FILE)
io_info = yaml.safe_load(open(IO_FILE, "r"))
exps = io_info["com_exp"]
for expid, exp in enumerate(exps):
    print(f"Experiment {expid}:")
    print("  ", " ".join([f"{k}: {v}" for k, v in exp.items()]))


def project_3d_points_to_2d(pts: np.ndarray, camera_params: dict) -> np.ndarray:
    """Project 3D points to a specific camera view

    Args:
        pts (_type_): 3D points [N, 3]
        camera_params (_type_): _description_

    Returns:
        np.ndarray: 2D points [N, 2]
    """
    projpts = project_to_2d(
        pts, camera_params["K"], camera_params["r"], camera_params["t"]
    )[:, :2]
    projpts = distortPoints(
        projpts,
        camera_params["K"],
        np.squeeze(camera_params["RDistort"]),
        np.squeeze(camera_params["TDistort"]),
    )
    return projpts.T


def visualize_experiment_files(exp: dict, n_cols: int = 3):
    label3d_file = exp["label3d_file"]
    # add to absolute dataroot
    label3d_file = os.path.join(dataroot, label3d_file)
    
    assert os.path.exists(label3d_file), f"label3d_file {label3d_file} does not exist"
    
    viddir = os.path.join(os.path.dirname(label3d_file), 'videos')
    
    # load label3d labels
    labels = load_labels(label3d_file)
    print(type(labels), len(labels))
    camera_params = load_camera_params(label3d_file)
    camnames = load_camnames(label3d_file)
    camera_params = {f"Camera{i+1}": params for i, params in enumerate(camera_params)}

    # data_sampleIDs are UNIQUE identifiers for synchronizing multi-view data streams
    # data_frames, can vary across views due to acqusition delays
    # both are 0-indexed
    data_sampleID = np.squeeze(labels[0]["data_sampleID"])
    print(f"Label3D file {label3d_file} contains {len(data_sampleID)} samples")

    com3d = labels[0]['data_3d'][:,0,:] # [N,3]
    # print(labels[0]['data_3d'].shape)
    # quit()
    
    # com_data = sio.loadmat(com_file)
    # labeled_indices = np.array([np.where(com_data["sampleID"][0] == sample) for sample in data_sampleID])
    # com3d = com_data["com"][labeled_indices]
    # pose3d = labels[0]["data_3d"]
    
    # load video readers
    vidreaders = {
        camname: imageio.get_reader(os.path.join(viddir, camname, "0.mp4"))
        for camname in camnames
    }

    # visualize a random sample
    vis_idx = np.random.randint(len(data_sampleID))
    
    n_rows = len(camnames) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    # skeleton = load_body_profile("rat23")
    
    for camidx, camname in enumerate(camnames):
        ax = axes[camidx]
        ax.set_title(camname)
        
        # retrieve the corresponding frame
        data_frames = np.squeeze(labels[camidx]["data_frame"])
        frame_idx = data_frames[vis_idx]
        
        image = vidreaders[camname].get_data(frame_idx)
        ax.imshow(image)
        
        # visualize 3d COMs
        _com3d = com3d[vis_idx].reshape((-1, 3))
        com2d = project_3d_points_to_2d(_com3d, camera_params[camname])
        ax.scatter(*com2d.T, s=10, c="r", label="COM-GT", marker="x")
        
        # visualize 3d pose
        # _pose3d = pose3d[vis_idx].reshape((-1, 3))
        # pose2d = project_3d_points_to_2d(_pose3d, camera_params[camname])

        # for limb in skeleton["limbs"]:
        #     ax.plot(pose2d[limb, 0], pose2d[limb, 1],
        #             c="#7fcdbb", linewidth=1, label="Pose-GT",
        #             marker="o", markersize=3, markerfacecolor="w", markeredgewidth=0.5, markeredgecolor="k",
        #     )
    plt.savefig('temp.pdf')
    return camera_params, None, com3d



camera_params, pose3d, com3d = visualize_experiment_files(exps[0])