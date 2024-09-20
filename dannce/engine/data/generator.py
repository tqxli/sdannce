"""Generator module for dannce training.
"""
import os
from copy import deepcopy
import numpy as np

from dannce.engine.data import ops
from dannce.engine.data.video import LoadVideoFrame
from dannce.engine.data.ops import Camera
import dannce.engine.utils.image as image_utils
import dannce.engine.utils.segmentation as seg_utils
import warnings
import time
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Dict, Tuple, Text
import cv2
import imageio

import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches

_DEFAULT_CAM_NAMES = [
    "CameraR",
    "CameraL",
    "CameraU",
    "CameraU2",
    "CameraS",
    "CameraE",
]
_EXEP_MSG = "Desired Label channels and ground truth channels do not agree"

MISSING_KEYPOINTS_MSG = (
    "If mirror augmentation is used, the right_keypoints indices and left_keypoints "
    + "indices must be specified as well. "
    + "For the skeleton, ['RHand, 'LHand', 'RFoot', 'LFoot'], "
    + "set right_keypoints: [0, 2] and left_keypoints: [1, 3] in the config file"
)

"""
Notes: 
For data generation, we separate video data acquisition from subsequent augmentation,
as it is impractical to repeat frame reading as the training progresses.

DataGenerator:
    Baseclass.

DataGenerator_3Dconv: 
    return BATCHED volumes without augmentation.

DataGenerator_3Dconv_social:
    return different volumes associated with the same frame. 
"""


class DataGenerator(torch.utils.data.Dataset):
    """
    Attributes:
        batch_size (int): Batch size to generate
        camnames (List): List of camera names.
        clusterIDs (List): List of sampleIDs
        crop_height (Tuple): (first, last) pixels in image height
        crop_width (tuple): (first, last) pixels in image width
        currvideo (Dict): Contains open video objects
        currvideo_name (Dict): Contains open video object names
        dim_in (Tuple): Input dimension
        dim_out (Tuple): Output dimension
        extension (Text): Video extension
        indexes (np.ndarray): sample indices used for batch generation
        labels (Dict): Label dictionary
        list_IDs (List): List of sampleIDs
        mono (bool): If True, use grayscale image.
        n_channels_in (int): Number of input channels
        n_channels_out (int): Number of output channels
        out_scale (int): Scale of the output gaussians.
        samples_per_cluster (int): Samples per cluster
        shuffle (bool): If True, shuffle the samples.
        vidreaders (Dict): Dict containing video readers.
        predict_flag (bool): If True, use imageio for reading videos, rather than OpenCV
    """

    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        clusterIDs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: float = 5,
        shuffle: bool = True,
        camnames: List = [],
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        samples_per_cluster: int = 0,
        vidreaders: Dict = None,
        chunks: int = 3500,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
    ):
        """Initialize Generator.
        """
        self.dim_in = dim_in
        self.dim_out = dim_in
        self.batch_size = batch_size
        self.labels = labels
        self.vidreaders = vidreaders
        self.list_IDs = list_IDs
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        # sigma for the ground truth joint probability map Gaussians
        self.out_scale = out_scale
        self.camnames = camnames
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.clusterIDs = clusterIDs
        self.samples_per_cluster = samples_per_cluster
        self._N_VIDEO_FRAMES = chunks
        self.mono = mono
        self.mirror = mirror
        self.predict_flag = predict_flag

        if self.vidreaders is not None:
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        assert len(self.list_IDs) == len(self.clusterIDs)

        self.load_frame = LoadVideoFrame(
            self._N_VIDEO_FRAMES, self.vidreaders, self.camnames, self.predict_flag
        )

    def __len__(self) -> int:
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))


class DataGenerator_3Dconv(DataGenerator):
    """Update generator class to handle multiple experiments.
    """

    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        labels_3d: Dict,
        camera_params: Dict,
        clusterIDs: List,
        com3d: Dict,
        tifdirs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: int = 5,
        shuffle: bool = True,
        camnames: List = [],
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        vmin: int = -100,
        vmax: int = 100,
        nvox: int = 32,
        gpu_id: Text = "0",
        interp: Text = "linear",
        depth: bool = False,
        channel_combo=None,
        mode: Text = "3dprob",
        samples_per_cluster: int = 0,
        immode: Text = "tif",
        rotation: bool = False,
        vidreaders: Dict = None,
        distort: bool = True,
        expval: bool = False,
        multicam: bool = True,
        var_reg: bool = False,
        COM_aug: bool = None,
        crop_im: bool = True,
        norm_im: bool = True,
        chunks: int = 3500,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
        segmentation_model=None,
    ):
        """Initialize data generator.

        Args:
            list_IDs (List): List of sample Ids
            labels (Dict): Dictionary of labels
            labels_3d (Dict): Dictionary of 3d labels.
            camera_params (Dict): Camera parameters dictionary.
            clusterIDs (List): List of sample Ids
            com3d (Dict): Dictionary of com3d data.
            tifdirs (List): Directories of .tifs
            batch_size (int, optional): Batch size to generate
            dim_in (Tuple, optional): Input dimension
            n_channels_in (int, optional): Number of input channels
            n_channels_out (int, optional): Number of output channels
            out_scale (int, optional): Scale of the output gaussians.
            shuffle (bool, optional): If True, shuffle the samples.
            camnames (List, optional): List of camera names.
            crop_width (Tuple, optional): (first, last) pixels in image width
            crop_height (Tuple, optional): (first, last) pixels in image height
            vmin (int, optional): Minimum box dim (relative to the COM)
            vmax (int, optional): Maximum box dim (relative to the COM)
            nvox (int, optional): Number of voxels per box side
            gpu_id (Text, optional): Identity of GPU to use.
            interp (Text, optional): Interpolation method.
            depth (bool): If True, appends voxel depth to sampled image features [DEPRECATED]
            channel_combo (Text): Method for shuffling camera input order
            mode (Text): Toggles output label format to match MAX vs. AVG network requirements.
            samples_per_cluster (int, optional): Samples per cluster
            immode (Text): Toggles using 'video' or 'tif' files as image input [DEPRECATED]
            rotation (bool, optional): If True, use simple rotation augmentation.
            vidreaders (Dict, optional): Dict containing video readers.
            distort (bool, optional): If true, apply camera undistortion.
            expval (bool, optional): If True, process an expected value network (AVG)
            multicam (bool): If True, formats data to work with multiple cameras as input.
            var_reg (bool): If True, adds a variance regularization term to the loss function.
            COM_aug (bool, optional): If True, augment the COM.
            crop_im (bool, optional): If True, crop images.
            norm_im (bool, optional): If True, normalize images.
            chunks (int, optional): Size of chunks when using chunked mp4.
            mono (bool, optional): If True, use grayscale image.
            predict_flag (bool, optional): If True, use imageio for reading videos, rather than OpenCV
        """
        DataGenerator.__init__(
            self,
            list_IDs,
            labels,
            clusterIDs,
            batch_size,
            dim_in,
            n_channels_in,
            n_channels_out,
            out_scale,
            shuffle,
            camnames,
            crop_width,
            crop_height,
            samples_per_cluster,
            vidreaders,
            chunks,
            mono,
            mirror,
            predict_flag,
        )
        self.vmin = vmin
        self.vmax = vmax
        self.nvox = nvox
        self.vsize = (vmax - vmin) / nvox
        self.dim_out_3d = (nvox, nvox, nvox)
        self.labels_3d = labels_3d
        self.camera_params = camera_params
        self.interp = interp
        self.depth = depth
        self.channel_combo = channel_combo
        print(self.channel_combo)
        self.gpu_id = gpu_id
        self.mode = mode
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im

        self.device = torch.device("cuda:" + self.gpu_id)

        self.threadpool = ThreadPool(len(self.camnames[0]))
        self.segmentation_model = segmentation_model

        ts = time.time()
        for experimentID in self.camnames.keys():
            for camname in self.camnames[experimentID]:
                # M only needs to be computed once for each camera
                K = self.camera_params[experimentID][camname]["K"]
                R = self.camera_params[experimentID][camname]["R"]
                t = self.camera_params[experimentID][camname]["t"]
                M = torch.as_tensor(ops.camera_matrix(K, R, t), dtype=torch.float32)
                self.camera_params[experimentID][camname]["M"] = M

        print("Init took {} sec.".format(time.time() - ts))

        self.pj_method = self.pj_grid_mirror if self.mirror else self.pj_grid

    def close_all_readers(self):
        self.threadpool.close()
        
        for vid in self.load_frame.currvideo.values():
            if isinstance(vid, imageio.core.Format.Reader):
                vid.close()

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y
                (np.ndarray): Target
        """
        # Find list of IDs
        list_IDs_temp = self.list_IDs[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_batched(self, indices):
        """Return batched inputs. Used during sdannce inference.

        Args:
            indices (list): list of frames indices
        """
        all_X = []
        for index in indices:
            X, y = self.__getitem__(index)
            all_X.append(X)

        n_inputs = len(all_X[0])
        batched_inputs = []
        for input_idx in range(n_inputs):
            batched_input = np.concatenate([X[input_idx] for X in all_X], axis=0)
            batched_inputs.append(batched_input)
        return batched_inputs

    def pj_grid(self, X_grid, camname, ID, experimentID):
        """Projects 3D voxel centers and sample images as projected 2D pixel coordinates

        Args:
            X_grid (np.ndarray): 3-D array containing center coordinates of each voxel.
            camname (Text): camera name
            ID (Text): string denoting a sample ID
            experimentID (int): identifier for a video recording session.

        Returns:
            np.ndarray: projected voxel centers, now in 2D pixels
        """
        ts = time.time()
        # Need this copy so that this_y does not change
        this_y = torch.as_tensor(
            self.labels[ID]["data"][camname], dtype=torch.float32, device=self.device
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * float("nan")
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, axis=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, axis=1)

        thisim = self.load_frame.load_vid_frame(
            self.labels[ID]["frames"][camname], camname, extension=self.extension,
        )[
            self.crop_height[0] : self.crop_height[1],
            self.crop_width[0] : self.crop_width[1],
        ]
        return self.pj_grid_post(
            X_grid, camname, ID, experimentID, com, com_precrop, thisim
        )

    def pj_grid_mirror(self, X_grid, camname, ID, experimentID, thisim):
        this_y = torch.as_tensor(
            self.labels[ID]["data"][camname], dtype=torch.float32, device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * float("nan")
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, dim=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, dim=1)

        if not self.mirror:
            raise Exception(
                "Trying to project onto mirrored images without mirror being set properly"
            )

        if self.camera_params[experimentID][camname]["m"] == 1:
            passim = thisim[-1::-1].copy()
        elif self.camera_params[experimentID][camname]["m"] == 0:
            passim = thisim.copy()
        else:
            raise Exception("Invalid mirror parameter, m, must be 0 or 1")

        return self.pj_grid_post(
            X_grid, camname, ID, experimentID, com, com_precrop, passim
        )

    def pj_grid_post(self, X_grid, camname, ID, experimentID, com, com_precrop, thisim):
        # separate the porjection and sampling into its own function so that
        # when mirror == True, this can be called directly
        if self.crop_im:
            if torch.all(torch.isnan(com)):
                thisim = torch.zeros(
                    (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                    dtype=torch.uint8,
                    device=self.device,
                )
            else:
                thisim, _ = image_utils.cropcom(thisim, com, size=self.dim_in[0])
        # print('Frame loading took {} sec.'.format(time.time() - ts))

        if self.segmentation_model is not None:
            input = [
                torchvision.transforms.functional.to_tensor(thisim.copy()).to(
                    self.device, dtype=torch.float
                )
            ]
            prediction = self.segmentation_model(input)[0]

            mask = prediction["masks"][0].permute(1, 2, 0).detach().cpu().numpy()
            mask = (mask >= 0.5).astype(np.uint8)

            # bbox = prediction["boxes"][0].cpu().numpy()
            # com_pred = ((bbox[0] + bbox[2]) / 2, (bbox[1]+bbox[3]) / 2)

            thisim *= mask  # return the segmented foreground object

        ts = time.time()
        proj_grid = ops.project_to2d(
            X_grid, self.camera_params[experimentID][camname]["M"], self.device
        )
        # print('Project2d took {} sec.'.format(time.time() - ts))

        ts = time.time()
        if self.distort:
            proj_grid = ops.distortPoints(
                proj_grid[:, :2],
                self.camera_params[experimentID][camname]["K"],
                np.squeeze(self.camera_params[experimentID][camname]["RDistort"]),
                np.squeeze(self.camera_params[experimentID][camname]["TDistort"]),
                self.device,
            )
            proj_grid = proj_grid.transpose(0, 1)
            # print('Distort took {} sec.'.format(time.time() - ts))

        ts = time.time()
        if self.crop_im:
            proj_grid = proj_grid[:, :2] - com_precrop + self.dim_in[0] // 2
            # Now all coordinates should map properly to the image cropped around the COM
        else:
            # Then the only thing we need to correct for is crops at the borders
            proj_grid = proj_grid[:, :2]
            proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
            proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

        rgb = ops.sample_grid(thisim, proj_grid, self.device, method=self.interp)
        # print('Sample grid {} sec.'.format(time.time() - ts))

        if (
            ~torch.any(torch.isnan(com_precrop))
            or (self.channel_combo == "avg")
            or not self.crop_im
        ):
            X = rgb.permute(0, 2, 3, 4, 1)

        return X

    def _init_vars(self, first_exp):
        X = torch.zeros(
            (
                self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d,
                self.n_channels_in + self.depth,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

        if self.mode == "3dprob":
            y_3d = torch.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype=torch.float32,
                device=self.device,
            )
        elif self.mode == "coordinates":
            y_3d = torch.zeros(
                (self.batch_size, 3, self.n_channels_out),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise Exception("not a valid generator mode")

        sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
        X_grid = torch.zeros(
            (self.batch_size, sz, 3), dtype=torch.float32, device=self.device,
        )

        return X, y_3d, X_grid

    def _generate_coord_grid(self, this_COM_3d):
        xgrid = torch.arange(
            self.vmin + this_COM_3d[0] + self.vsize / 2,
            this_COM_3d[0] + self.vmax,
            self.vsize,
            dtype=torch.float32,
            device=self.device,
        )
        ygrid = torch.arange(
            self.vmin + this_COM_3d[1] + self.vsize / 2,
            this_COM_3d[1] + self.vmax,
            self.vsize,
            dtype=torch.float32,
            device=self.device,
        )
        zgrid = torch.arange(
            self.vmin + this_COM_3d[2] + self.vsize / 2,
            this_COM_3d[2] + self.vmax,
            self.vsize,
            dtype=torch.float32,
            device=self.device,
        )
        (x_coord_3d, y_coord_3d, z_coord_3d) = torch.meshgrid(xgrid, ygrid, zgrid)

        # torch.meshgrid current behavior is the same as np.meshgrid('ij')
        # need to convert to 'xy' for generating correct Gaussian targets
        x_coord_3d, y_coord_3d, z_coord_3d = (
            x_coord_3d.transpose(0, 1),
            y_coord_3d.transpose(0, 1),
            z_coord_3d.transpose(0, 1),
        )

        grid = torch.stack(
            (x_coord_3d.flatten(), y_coord_3d.flatten(), z_coord_3d.flatten()), dim=1
        )

        return (x_coord_3d, y_coord_3d, z_coord_3d), grid

    def _generate_targets(self, i, y_3d, this_y_3d, coords_3d):
        if self.mode == "3dprob":
            # generate Gaussian targets
            for j in range(self.n_channels_out):
                y_3d[i, j] = torch.exp(
                    -(
                        (coords_3d[1] - this_y_3d[1, j]) ** 2
                        + (coords_3d[0] - this_y_3d[0, j]) ** 2
                        + (coords_3d[2] - this_y_3d[2, j]) ** 2
                    )
                    / (2 * self.out_scale ** 2)
                )
                # When the voxel grid is coarse, we will likely miss
                # the peak of the probability distribution, as it
                # will lie somewhere in the middle of a large voxel.
                # So here we renormalize to [~, 1]

        if self.mode == "coordinates":
            if this_y_3d.shape == y_3d[i].shape:
                y_3d[i] = this_y_3d
            else:
                msg = "Note: ignoring dimension mismatch in 3D labels"
                warnings.warn(msg)

        return y_3d

    def _adjust_vol_channels(self, X, y_3d, first_exp, num_cams):
        if self.multicam:
            X = X.reshape(
                (
                    self.batch_size,
                    len(self.camnames[first_exp]),
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    X.shape[4],
                )
            )
            X = X.permute(0, 2, 3, 4, 5, 1)

            if self.channel_combo == "avg":
                X = torch.mean(X, dim=-1)

            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == "random":
                X = X[..., torch.randperm(X.shape[-1])]
                X = X.transpose(4, 5).reshape(*X.shape[:4], -1)
            else:
                X = X.transpose(4, 5).reshape(*X.shape[:4], -1)
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = y_3d.repeat(num_cams, 1, 1, 1, 1)

        return X, y_3d

    def _convert_tensor_to_numpy(self, X, y_3d, X_grid):
        # ts = time.time()
        if torch.is_tensor(X):
            X = X.float().cpu().numpy()
        if torch.is_tensor(y_3d):
            y_3d = y_3d.cpu().numpy()
        if torch.is_tensor(X_grid):
            X_grid = X_grid.cpu().numpy()
        # print('Numpy took {} sec'.format(time.time() - ts))

        return X, y_3d, X_grid

    def _finalize_samples(self, X, y_3d, X_grid):
        if self.var_reg or self.norm_im:
            X = image_utils.preprocess_3d(X)

        inputs, targets = [X, X_grid], [y_3d]

        # if self.expval:
        #     inputs.append(X_grid)

        if self.var_reg:
            targets.append(torch.zeros((self.batch_size, 1)))

        return inputs, targets

    def __data_generation(self, list_IDs_temp):
        """
        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])
        X, y_3d, X_grid = self._init_vars(first_exp)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            num_cams = len(self.camnames[experimentID])

            # For 3D ground truth (keypoints, COM)
            this_y_3d = torch.as_tensor(
                self.labels_3d[ID], dtype=torch.float32, device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], dtype=torch.float32, device=self.device
            )

            # Create and project the grid here,
            coords_3d, grid = self._generate_coord_grid(this_COM_3d)
            X_grid[i] = grid

            # Generate training targets
            y_3d = self._generate_targets(i, y_3d, this_y_3d, coords_3d)

            # Compute projected images in parallel using multithreading
            # ts = time.time()
            arglist = []
            if self.mirror:
                # Here we only load the video once, and then parallelize the projection
                # and sampling after mirror flipping. For setups that collect views
                # in a single image with the use of mirrors
                loadim = self.load_frame.load_vid_frame(
                    self.labels[ID]["frames"][self.camnames[experimentID][0]],
                    self.camnames[experimentID][0],
                    extension=self.extension,
                )[
                    self.crop_height[0] : self.crop_height[1],
                    self.crop_width[0] : self.crop_width[1],
                ]

            for c in range(num_cams):
                args = [X_grid[i], self.camnames[experimentID][c], ID, experimentID]
                if self.mirror:
                    args.append(loadim)

                arglist.append(args)

            result = self.threadpool.starmap(self.pj_method, arglist)

            for c in range(num_cams):
                ic = c + i * num_cams
                X[ic, :, :, :, :] = result[c]
            # print('MP took {} sec.'.format(time.time()-ts))

        # adjust volume channels
        X, y_3d = self._adjust_vol_channels(X, y_3d, first_exp, num_cams)

        # 3dprob is required for *training* MAX networks
        if self.mode == "3dprob":
            y_3d = y_3d.permute([0, 2, 3, 4, 1])

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = X.reshape(*X.shape[:4], len(self.camnames[first_exp]), -1)
            X = X[..., 0] * 0.2125 + X[..., 1] * 0.7154 + X[..., 2] * 0.0721

        # Convert pytorch tensors back to numpy array
        X, y_3d, X_grid = self._convert_tensor_to_numpy(X, y_3d, X_grid)

        return self._finalize_samples(X, y_3d, X_grid)


class DataGenerator_3Dconv_social(DataGenerator_3Dconv):
    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        labels_3d: Dict,
        camera_params: Dict,
        clusterIDs: List,
        com3d: Dict,
        tifdirs: List,
        n_instances=2,
        occlusion=False,
        **kwargs,
    ):
        DataGenerator_3Dconv.__init__(
            self,
            list_IDs,
            labels,
            labels_3d,
            camera_params,
            clusterIDs,
            com3d,
            tifdirs,
            **kwargs,
        )

        self.n_instances = n_instances
        self.batch_size = n_instances
        self.occlusion = occlusion
        self.list_IDs = [
            ID for ID in self.list_IDs if int(ID.split("_")[0]) % n_instances == 0
        ]

    def __len__(self) -> int:
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        return len(self.list_IDs)

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y
                (np.ndarray): Target
        """
        # Find list of IDs
        thisID = self.list_IDs[index]
        experimentID, sampleID = thisID.split("_")

        list_IDs_temp = [
            str(int(experimentID) + i) + "_" + sampleID for i in range(self.n_instances)
        ]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_batched(self, indices):
        """Return batched inputs. Used during sdannce inference.

        Args:
            indices (list): list of frames indices
        """
        all_X = []
        for index in indices:
            X, y = self.__getitem__(index)
            all_X.append(X)

        n_inputs = len(all_X[0])
        batched_inputs = []
        for input_idx in range(n_inputs):
            batched_input = np.concatenate([X[input_idx] for X in all_X], axis=0)
            batched_inputs.append(batched_input)
        return batched_inputs

    def proj_grid(self, X_grids, camnames, IDs, experimentIDs, com_3ds):
        """Projects 3D voxel centers and sample images as projected 2D pixel coordinates

        Args:
            X_grid (np.ndarray): 3-D array containing center coordinates of each voxel.
            camname (Text): camera name
            ID (Text): string denoting a sample ID
            experimentID (int): identifier for a video recording session.

        Returns:
            np.ndarray: projected voxel centers, now in 2D pixels
        """
        ts = time.time()

        # Need this copy so that this_y does not change
        thisims, coms, com_precrops = [], [], []

        # only load the frame once for all animals present
        thisim = self.load_frame.load_vid_frame(
            self.labels[IDs[0]]["frames"][camnames[0]],
            camnames[0],
            extension=self.extension,
        )[
            self.crop_height[0] : self.crop_height[1],
            self.crop_width[0] : self.crop_width[1],
        ]

        for i in range(self.n_instances):
            this_y = torch.as_tensor(
                self.labels[IDs[i]]["data"][camnames[i]],
                dtype=torch.float32,
                device=self.device,
            ).round()

            if torch.all(torch.isnan(this_y)):
                com_precrop = torch.zeros_like(this_y[:, 0]) * float("nan")
            else:
                com_precrop = torch.mean(this_y, axis=1)

            this_y[0, :] = this_y[0, :] - self.crop_width[0]
            this_y[1, :] = this_y[1, :] - self.crop_height[0]
            com = torch.mean(this_y, axis=1)

            coms.append(com)
            com_precrops.append(com_precrop)
            thisims.append(thisim)

        return self.pj_grid_post(
            X_grids, camnames, IDs, experimentIDs, coms, com_precrops, thisims, com_3ds
        )

    def pj_grid_mirror(self, X_grid, camname, ID, experimentID, thisim):
        this_y = torch.as_tensor(
            self.labels[ID]["data"][camname], dtype=torch.float32, device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * float("nan")
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, dim=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, dim=1)

        if not self.mirror:
            raise Exception(
                "Trying to project onto mirrored images without mirror being set properly"
            )

        if self.camera_params[experimentID][camname]["m"] == 1:
            passim = thisim[-1::-1].copy()
        elif self.camera_params[experimentID][camname]["m"] == 0:
            passim = thisim.copy()
        else:
            raise Exception("Invalid mirror parameter, m, must be 0 or 1")

        return self.pj_grid_post(
            X_grid, camname, ID, experimentID, com, com_precrop, passim
        )

    def visualize_2d(
        self,
        im,
        IDs,
        camnames,
        camcoords,
        bb1,
        bb2,
        scores,
        savedir="./vis_occlusion/2021_07_03_M2_M3",
    ):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        depths = camcoords[:, 2]
        fname = IDs[0].split("_")[-1] + "_" + camnames[0].split("_")[-1] + ".jpg"
        fig, ax = plt.subplots(1, 1)
        ax.imshow(im)
        ax.set_title(
            "B: {:.2f}, {:.2f} | R: {:.2f}, {:.2f}".format(
                depths[0], scores[0], depths[1], scores[1]
            )
        )
        # Create a Rectangle patch
        rect1 = patches.Rectangle(
            (bb1[0], bb1[1]),
            bb1[2] - bb1[0],
            bb1[3] - bb1[1],
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        rect2 = patches.Rectangle(
            (bb2[0], bb2[1]),
            bb2[2] - bb2[0],
            bb2[3] - bb2[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        fig.savefig(os.path.join(savedir, fname))
        plt.close(fig)

    @classmethod
    def apply_mask(self, image, mask, color, alpha=0.3):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(
                mask == 1,
                image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                image[:, :, c],
            )
        return image

    def visualize_mask(
        self, IDs, camnames, im, masks, coms, msg, savedir="./vis_occlusion_masks"
    ):
        colors = [(0, 255, 125), (252, 51, 51)]
        if not os.path.exists(savedir):
            print("Saving to ", savedir)
            os.makedirs(savedir)
        fname = IDs[0].split("_")[-1] + "_" + camnames[0].split("_")[-1] + ".jpg"

        fig, ax = plt.subplots(1, 1)
        for i, mask in enumerate(masks):
            im = self.apply_mask(im, np.squeeze(mask), colors[i])
        ax.imshow(im)

        com1 = coms[0]
        com2 = coms[1]
        ax.scatter(x=com1[0], y=com1[1], color="blue")
        ax.scatter(x=com2[0], y=com2[1], color="red")

        ax.set_title(msg)

        fig.savefig(os.path.join(savedir, fname))
        plt.close(fig)

    def pj_grid_post(
        self,
        X_grids,
        camnames,
        IDs,
        experimentIDs,
        coms,
        com_precrops,
        thisims,
        com_3ds,
    ):
        # separate the porjection and sampling into its own function so that
        # when mirror == True, this can be called directly
        if self.crop_im:
            for i in range(self.n_instances):
                if torch.all(torch.isnan(coms[i])):
                    thisims[i] = torch.zeros(
                        (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                else:
                    thisims[i] = image_utils.cropcom(
                        thisims[i], coms[i], size=self.dim_in[0]
                    )[0]
        # print('Frame loading took {} sec.'.format(time.time() - ts))

        # convert 3D COMs into camera coordinate frame for occlusion check
        com_3ds_cam = (
            ops.world_to_cam(
                com_3ds.clone(),
                self.camera_params[experimentIDs[0]][camnames[0]]["M"],
                self.device,
            )
            .detach()
            .cpu()
            .numpy()
        )
        depths = com_3ds_cam[:, 2]
        com_2ds_pix = com_3ds_cam[:, :2] / com_3ds_cam[:, 2:]

        # generate bounding boxes in pixel coordinates
        # this only APPROXIMATES the occlusion between different animals
        w, h = 200, 150
        bb1 = [
            com_2ds_pix[0, 0] - w,
            com_2ds_pix[0, 1] - h,
            com_2ds_pix[0, 0] + w,
            com_2ds_pix[0, 1] + h,
        ]
        bb2 = [
            com_2ds_pix[1, 0] - w,
            com_2ds_pix[1, 1] - h,
            com_2ds_pix[1, 0] + w,
            com_2ds_pix[1, 1] + h,
        ]

        # check depths
        instance_front, instance_back = np.argmin(depths), np.argmax(depths)
        occlusion_scores = np.zeros(
            (self.n_instances)
        )  # the foreground animal is not occluded
        # check overlap region
        occlusion_scores[instance_back] = seg_utils.bbox_iou(bb1, bb2)

        # self.visualize_2d(thisims[0], IDs, camnames, com_3ds_cam, bb1, bb2, occlusion_scores)

        if self.segmentation_model is not None:
            # if there is no occlusion,
            # the model should be able to detect `n_instances`` masks
            # in this specific camera view, with high confidence (score > 0.9)
            # occlusion_flag = (occlusion_scores[instance_front] > 0.8)
            # initialize masks as ones ==> no influence if no masks predicted
            masks = np.ones((self.n_instances, *thisims[0].shape[:2], 1))

            # get mask predictions
            input = [
                torchvision.transforms.functional.to_tensor(thisims[0].copy()).to(
                    self.device, dtype=torch.float
                )
            ]
            prediction = self.segmentation_model(input)[0]
            # filter by confidence scores
            filtering_by_scores = prediction["scores"] > 0.85
            raw_masks = prediction["masks"][filtering_by_scores]
            # print(f"{len(raw_masks)} masks detected.")
            if len(raw_masks) > self.n_instances:
                raw_masks = raw_masks[: self.n_instances]

            # adjust dimension, convert to numpy
            masks_unordered = []
            for j in range(len(raw_masks)):
                mask = raw_masks[j].permute(1, 2, 0).detach().cpu().numpy()
                masks_unordered.append((mask >= 0.5).astype(np.uint8))
            masks_unordered = np.stack(masks_unordered)

            # mask matching
            msg = ""
            if len(raw_masks) == 0:
                msg = "No mask predicted."

            elif len(raw_masks) < self.n_instances:
                counts = seg_utils.compute_support(com_2ds_pix, masks_unordered[0])
                assignment = np.argmax(counts)
                masks[assignment] = masks_unordered[0]

                if assignment == instance_front:
                    msg = "Mask only predicted on the foreground animal."
                    non_assignment = (np.arange(self.n_instances)) != assignment
                    # masks[non_assignment] = (masks_unordered[0] == 0).astype(np.uint8)
                else:
                    msg = "Mask only predicted on animal behind."

            elif len(raw_masks) == self.n_instances:
                # remove intersected region
                # mask_intersect = processing.mask_intersection(masks_unordered[0], masks_unordered[1])
                # masks_unordered = [mask-mask_intersect for mask in masks_unordered]

                counts0 = seg_utils.compute_support(com_2ds_pix, masks_unordered[0])
                counts1 = seg_utils.compute_support(com_2ds_pix, masks_unordered[1])

                assignment0 = np.argmax(counts0)
                assignment1 = np.argmax(counts1)

                if assignment0 != assignment1:
                    # perfect matching
                    msg = "Perfect matching."
                    masks[assignment0] = masks_unordered[0]
                    masks[assignment1] = masks_unordered[1]
                elif assignment0 == instance_front:
                    msg = "Mask ambiguity. Assume the higher confidence mask belongs to the front."
                    masks[instance_front] = masks_unordered[0]
                    masks[instance_back] = masks_unordered[1]
                else:
                    msg = "Mask ambiguity."

            # self.visualize_mask(IDs, camnames, thisims[0].copy(), masks, com_2ds_pix, msg)
            for i, im in enumerate(thisims):
                # thisims[i] = im * masks[i]
                thisims[i] = np.tile(masks[i], (1, 1, 3))

        X = []
        for i in range(self.n_instances):
            # ts = time.time()
            proj_grid = ops.project_to2d(
                X_grids[i],
                self.camera_params[experimentIDs[i]][camnames[i]]["M"],
                self.device,
            )
            # print('Project2d took {} sec.'.format(time.time() - ts))

            # ts = time.time()
            if self.distort:
                proj_grid = ops.distortPoints(
                    proj_grid[:, :2],
                    self.camera_params[experimentIDs[i]][camnames[i]]["K"],
                    np.squeeze(
                        self.camera_params[experimentIDs[i]][camnames[i]]["RDistort"]
                    ),
                    np.squeeze(
                        self.camera_params[experimentIDs[i]][camnames[i]]["TDistort"]
                    ),
                    self.device,
                )
                proj_grid = proj_grid.transpose(0, 1)
                # print('Distort took {} sec.'.format(time.time() - ts))

            # ts = time.time()
            if self.crop_im:
                proj_grid = proj_grid[:, :2] - com_precrops[i] + self.dim_in[0] // 2
                # Now all coordinates should map properly to the image cropped around the COM
            else:
                # Then the only thing we need to correct for is crops at the borders
                proj_grid = proj_grid[:, :2]
                proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
                proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

            rgb = ops.sample_grid(
                thisims[i], proj_grid, self.device, method=self.interp
            )

            if (
                ~torch.any(torch.isnan(com_precrops[i]))
                or (self.channel_combo == "avg")
                or not self.crop_im
            ):
                X.append(rgb.permute(0, 2, 3, 4, 1))

        return X, occlusion_scores

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
                rotangle: Rotation angle
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])

        X, y_3d, X_grid = self._init_vars(first_exp)

        com_3ds = torch.zeros(
            (self.batch_size, 3), dtype=torch.float32, device=self.device
        )

        # Generate data
        experimentIDs = []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            experimentIDs.append(experimentID)

            # For 3D ground truth
            this_y_3d = torch.as_tensor(
                self.labels_3d[ID], dtype=torch.float32, device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], dtype=torch.float32, device=self.device
            )

            com_3ds[i] = this_COM_3d

            # Create and project the grid here,
            coords_3d, grid = self._generate_coord_grid(this_COM_3d)
            X_grid[i] = grid

            # Generate training targets
            y_3d = self._generate_targets(i, y_3d, this_y_3d, coords_3d)

        # Compute projected images in parallel using multithreading
        ts = time.time()
        arglist = []

        num_cams = len(self.camnames[experimentIDs[0]])
        occlusion_scores = np.zeros((self.batch_size, num_cams), dtype=float)

        for c in range(num_cams):
            arglist.append(
                [
                    X_grid,
                    [self.camnames[experimentID][c] for experimentID in experimentIDs],
                    list_IDs_temp,
                    experimentIDs,
                    com_3ds,
                ]
            )
        result = self.threadpool.starmap(self.proj_grid, arglist)

        for c in range(num_cams):
            for j in range(self.n_instances):
                ic = c + j * num_cams
                X[ic, ...] = result[c][0][j][0]  # [H, W, D, C]
            occlusion_scores[:, c] = result[c][1]  # [2]
        # print('MP took {} sec.'.format(time.time()-ts))

        # adjust camera channels
        X, y_3d = self._adjust_vol_channels(X, y_3d, first_exp, num_cams)

        # 3dprob is required for *training* MAX networks
        if self.mode == "3dprob":
            y_3d = y_3d.permute([0, 2, 3, 4, 1])

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = X.reshape(*X.shape[:4], len(self.camnames[first_exp]), -1)
            X = X[..., 0] * 0.2125 + X[..., 1] * 0.7154 + X[..., 2] * 0.0721

        # Convert pytorch tensors back to numpy array
        X, y_3d, X_grid = self._convert_tensor_to_numpy(X, y_3d, X_grid)

        return self._finalize_samples(X, y_3d, X_grid, occlusion_scores)

    def _downscale_occluded_views(self, X, occlusion_scores):
        """
        X: [BS, H, W, D, n_cams*3]
        occlusion_scores: [BS, n_cams]
        """
        occluded_X = np.reshape(
            X.copy(), (*X.shape[:4], occlusion_scores.shape[-1], -1)
        )
        occlusion_scores = occlusion_scores[
            :, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis
        ]

        occluded_X *= occlusion_scores

        occluded_X = np.reshape(occluded_X, (*X.shape[:4], -1))

        return occluded_X

    def _finalize_samples(self, X, y_3d, X_grid, occlusion_scores):
        if self.var_reg or self.norm_im:
            X = image_utils.preprocess_3d(X)

        inputs, targets = [X], [y_3d]

        if self.expval:
            inputs.append(X_grid)

        if self.occlusion:
            # occluded_X = self._downscale_occluded_views(X, occlusion_scores)
            inputs.append(occlusion_scores)

        if self.var_reg:
            targets.append(torch.zeros((self.batch_size, 1)))

        return inputs, targets


class MultiviewImageGenerator(DataGenerator_3Dconv):
    def __init__(self, resize=True, image_size=256, crop=True, crop_size=512, **kwargs):

        super(MultiviewImageGenerator, self).__init__(**kwargs)
        self.image_size = image_size
        self.resize = resize
        self.crop = crop
        self.crop_size = crop_size
        self._get_camera_objs()

        self.ds_fac = self.crop_size / self.image_size

    def _get_camera_objs(self):
        self.camera_objs = {}
        for experimentID in self.camera_params.keys():
            self.camera_objs[experimentID] = {}

            for camname in self.camnames[experimentID]:
                param = self.camera_params[experimentID][camname]
                self.camera_objs[experimentID][camname] = Camera(
                    R=param["R"],
                    t=param["t"],
                    K=param["K"],
                    tdist=param["TDistort"],
                    rdist=param["RDistort"],
                )

    def _get_bbox(self, com_precrop):
        return (
            com_precrop[0] - self.crop_size // 2,
            com_precrop[1] - self.crop_size // 2,
            com_precrop[0] + self.crop_size // 2,
            com_precrop[1] + self.crop_size // 2,
        )

    def _project_to_views(self, ID, camname, experimentID):
        com3d = torch.from_numpy(self.com3d[ID]).unsqueeze(0).float()
        cam = self.camera_objs[experimentID][camname]
        proj2d = ops.project_to2d(com3d, cam.M, "cpu")[:, :2]
        proj2d = ops.distortPoints(proj2d, cam.K, cam.rdist, cam.tdist, "cpu")

        return proj2d.numpy()

    def _save_image_bbox(
        self,
        list_IDs,
        X,
        y_2d,
        bbox_offset=(20, 30, 20, 20),
        imdir="/media/mynewdrive/datasets/detection/images",
        annotdir="/media/mynewdrive/datasets/detection/annotations",
    ):
        if not os.path.exists(imdir):
            os.makedirs(imdir)
        if not os.path.exists(annotdir):
            os.makedirs(annotdir)

        for ID, ims, y in zip(list_IDs, X, y_2d):
            for cam in range(ims.shape[0]):
                fname = f"{ID}_cam{cam+1}"
                im = ims[cam].cpu().permute(1, 2, 0).numpy()

                # get tight bounding box from y_2d
                kpts = y[cam].cpu().numpy()
                x1, y1, x2, y2 = (
                    kpts[0].min(),
                    kpts[1].min(),
                    kpts[0].max(),
                    kpts[1].max(),
                )
                x1, x2 = x1 - bbox_offset[0], x2 + bbox_offset[2]
                y1, y2 = y1 - bbox_offset[1], y2 + bbox_offset[3]

                annot = {
                    "file_name": os.path.join(imdir, fname),
                    "bbox": [x1, y1, x2, y2],
                    "keypoints": kpts,
                    "category_id": 1,
                }

                # fig, ax = plt.subplots()
                # ax.imshow(im)

                # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, color='dodgerblue', fill=False)
                # ax.add_patch(rect)

                # ax.scatter(kpts[0], kpts[1], marker='.', color='r', linewidths=0.5)

                # fig.savefig(os.path.join(imdir, fname+".jpg"))
                cv2.imwrite(
                    os.path.join(imdir, fname + ".png"),
                    cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                )
                np.save(os.path.join(annotdir, fname), annot)

    def _visualize_multiview(self, ID, ims, y_2d, savedir="debug_image"):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        fig, axes = plt.subplots(2, 3, figsize=(10, 10))
        axes = axes.flatten()
        fname = f"{ID}.jpg"

        for i, ax in enumerate(axes):
            ax.imshow(ims[i].cpu().permute(1, 2, 0).numpy())

            # # Plot keypoints
            # if not np.isnan(y_2d).all():
            #     ax.scatter(y_2d[i, 0].cpu().numpy(), y_2d[i, 1].cpu().numpy(), marker='.', color='r', linewidths=0.5)

        fig.savefig(os.path.join(savedir, fname))
        plt.close(fig)

    def _load_im(self, ID, camname, experimentID):
        this_y = self.labels[ID]["data"][camname]
        com_precrop = self._project_to_views(ID, camname, experimentID)
        com_precrop = com_precrop.round().astype("float32")
        # com_precrop = np.nanmean(this_y.round(), axis=1).astype("float32")
        this_y = torch.tensor(this_y, dtype=torch.float32)

        im = self.load_frame.load_vid_frame(
            self.labels[ID]["frames"][camname], camname, extension=self.extension,
        )
        cam = deepcopy(self.camera_objs[experimentID][camname])
        new_y = this_y.clone()  # [2, 23]

        if self.crop:
            # crop images due to memory constraints
            im, cropdim = image_utils.cropcom(im, com_precrop, size=self.crop_size)
            bbox = self._get_bbox(com_precrop)
            cam.update_after_crop(bbox)
            new_y[0, :] -= cropdim[2]
            new_y[1, :] -= cropdim[0]

        # resize
        if self.resize:
            # old_height, old_width = im.shape[:2]
            # # im = cv2.resize(im, (self.image_size, self.image_size))
            # cam.update_after_resize((old_height, old_width), (self.image_size, self.image_size))
            # new_y[1, :] *= (im.shape[0] / old_height) # y
            # new_y[0, :] *= (im.shape[1] / old_width)  # x
            im = image_utils.downsample_batch(im[np.newaxis, ...], fac=self.ds_fac)
            im = np.squeeze(im)
            new_y /= self.ds_fac

        return im, cam, new_y

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y
                (np.ndarray): Target
        """
        # Find list of IDs
        list_IDs_temp = self.list_IDs[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])
        X, y_3d, X_grid = self._init_vars(first_exp)
        # X = X.new_zeros(
        #     (self.batch_size,
        #     len(self.camnames[first_exp]),
        #     3, 512, 512
        # ))
        X = []
        y_2d, cameras = [], []

        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            num_cams = len(self.camnames[experimentID])

            # For 3D ground truth (keypoints, COM)
            this_y_3d = torch.as_tensor(
                self.labels_3d[ID], dtype=torch.float32, device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], dtype=torch.float32, device=self.device
            )

            # Create and project the grid here,
            coords_3d, grid = self._generate_coord_grid(this_COM_3d)
            X_grid[i] = grid

            # Generate training targets
            y_3d = self._generate_targets(i, y_3d, this_y_3d, coords_3d)

            # extract multi-view images
            arglist = []
            for c in range(num_cams):
                arglist.append([ID, self.camnames[experimentID][c], experimentID])
            results = self.threadpool.starmap(self._load_im, arglist)

            ims = np.stack([r[0] for r in results], axis=0).astype(
                np.uint8
            )  # [6, H, W, 3]
            ims = torch.tensor(ims).permute(0, 3, 1, 2)
            X.append(ims)

            # also need camera params for each view
            cameras.append([r[1] for r in results])  # [BS, 6]

            # potentially need 2d labels as well
            y_2d.append(torch.stack([r[2] for r in results], dim=0))

        X = torch.stack(X, dim=0)
        try:
            y_2d = torch.stack(y_2d, dim=0)  # [BS, 6, 2, n_joints]
        except:
            y_2d = torch.zeros(X.shape[0], 6, 2, 1)
            print("Found {}".format(list_IDs_temp))

        # self._visualize_multiview(list_IDs_temp[0], X[0], y_2d[0])
        # breakpoint()
        # self._save_image_bbox(list_IDs_temp, X, y_2d)

        return (X.cpu(), X_grid.cpu(), cameras), (y_3d.cpu(), y_2d.cpu())


class DataGenerator_COM(torch.utils.data.Dataset):
    def __init__(
        self,
        n_instances,
        list_IDs,
        labels,
        vidreaders,
        batch_size=32,
        dim_in=(1024, 1280),
        n_channels_in=1,
        n_channels_out=1,
        out_scale=5,
        shuffle=True,
        camnames=_DEFAULT_CAM_NAMES,
        crop_width=(0, 1024),
        crop_height=(20, 1300),
        downsample=1,
        immode="video",
        labelmode="prob",
        dsmode="dsm",
        chunks=3500,
        multimode=False,
        mono=False,
        mirror=False,
        predict_flag=False,
    ):
        self.n_instances = n_instances
        self.dim_in = dim_in
        self.dim_out = dim_in
        self.batch_size = batch_size
        self.labels = labels
        self.vidreaders = vidreaders
        self.list_IDs = list_IDs
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        # sigma for the ground truth joint probability map Gaussians
        self.out_scale = out_scale
        self.camnames = camnames
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.downsample = downsample
        self.dsmode = dsmode
        self.on_epoch_end()

        if immode == "video":
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        self.immode = immode
        self.labelmode = labelmode
        # self.chunks = int(chunks)
        self.multimode = multimode

        self._N_VIDEO_FRAMES = chunks

        self.mono = mono

        self.predict_flag = predict_flag
        self.load_frame = LoadVideoFrame(
            self._N_VIDEO_FRAMES, self.vidreaders, self.camnames, self.predict_flag
        )

    def close_all_readers(self):
        for vid in self.load_frame.currvideo.values():
            if isinstance(vid, imageio.core.Format.Reader):
                vid.close()

    def __len__(self):
        """Denote the number of batches per epoch."""
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[index]]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_tif_frame(self, ind, camname):
        """Load frames in tif mode."""
        # In tif mode, vidreaders should just be paths to the tif directory
        return imageio.imread(
            os.path.join(self.vidreaders[camname], "{}.tif".format(ind))
        )

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        # X : (n_samples, *dim, n_channels)
        """
        # Initialization
        X = np.empty(
            (
                self.batch_size * len(self.camnames[0]),
                *self.dim_in,
                self.n_channels_in,
            ),
            dtype="uint8",
        )

        # We'll need to transpose this later such that channels are last,
        # but initializaing the array this ways gives us
        # more flexibility in terms of user-defined array sizes\
        if self.labelmode == "prob":
            y = np.empty(
                (
                    self.batch_size * len(self.camnames[0]),
                    self.n_channels_out,
                    *self.dim_out,
                ),
                dtype="float32",
            )
        else:
            # Just return the targets, without making a meshgrid later
            y = np.empty(
                (
                    self.batch_size * len(self.camnames[0]),
                    self.n_channels_out,
                    len(self.dim_out),
                ),
                dtype="float32",
            )

        # Generate data
        cnt = 0
        for i, ID in enumerate(list_IDs_temp):
            if "_" in ID:
                experimentID = int(ID.split("_")[0])
            else:
                # Then we only have one experiment
                experimentID = 0
            for camname in self.camnames[experimentID]:
                # Store sample
                # TODO(Refactor): This section is tricky to read
                if self.immode == "video":
                    X[cnt] = self.load_frame.load_vid_frame(
                        self.labels[ID]["frames"][camname], camname, self.extension,
                    )[
                        self.crop_height[0] : self.crop_height[1],
                        self.crop_width[0] : self.crop_width[1],
                    ]
                elif self.immode == "tif":
                    X[cnt] = self.load_tif_frame(
                        self.labels[ID]["frames"][camname], camname
                    )[
                        self.crop_height[0] : self.crop_height[1],
                        self.crop_width[0] : self.crop_width[1],
                    ]
                else:
                    raise Exception("Not a valid image reading mode")

                # Labels will now be the pixel positions of each joint.
                # Here, we convert them to
                # probability maps with a numpy meshgrid operation
                this_y = np.round(self.labels[ID]["data"][camname])
                if self.immode == "video":
                    this_y[0, :] = this_y[0, :] - self.crop_width[0]
                    this_y[1, :] = this_y[1, :] - self.crop_height[0]
                else:
                    raise Exception(
                        "Unsupported image format. Needs to be video files."
                    )
                # import pdb
                # pdb.set_trace()
                if self.labelmode == "prob":
                    (x_coord, y_coord) = np.meshgrid(
                        np.arange(self.dim_out[1]), np.arange(self.dim_out[0])
                    )

                    # Get the probability maps for all instances
                    instance_prob = []
                    for instance in range(self.n_instances):
                        instance_prob.append(
                            np.exp(
                                -(
                                    (y_coord - this_y[1, instance]) ** 2
                                    + (x_coord - this_y[0, instance]) ** 2
                                )
                                / (2 * self.out_scale ** 2)
                            )
                        )

                    # If using single-channel multi_instance take the max
                    # across probability maps. Otherwise assign a probability
                    # map to each channel.
                    if self.n_channels_out == 1:
                        y[cnt, 0] = np.max(np.stack(instance_prob, axis=2), axis=2)
                    else:
                        if len(instance_prob) != self.n_channels_out:
                            raise ValueError(
                                "n_channels_out != n_instances. This is necessary for multi-channel multi-instance tracking."
                            )
                        for j, instance in enumerate(instance_prob):
                            y[cnt, j] = instance
                else:
                    y[cnt] = this_y.T
                # plt.imshow(np.squeeze(y[0,:,:,:]))
                # plt.show()
                # import pdb
                # pdb.set_trace()
                cnt = cnt + 1

        # Move channels last
        if self.labelmode == "prob":
            y = np.transpose(y, [0, 2, 3, 1])
        else:
            # One less dimension when not training with probability map targets
            y = np.transpose(y, [0, 2, 1])

        if self.downsample > 1:
            X = image_utils.downsample_batch(X, fac=self.downsample, method=self.dsmode)
            if self.labelmode == "prob":
                y = image_utils.downsample_batch(
                    y, fac=self.downsample, method=self.dsmode
                )
                y /= np.max(np.max(y, axis=1), axis=1)[:, np.newaxis, np.newaxis, :]

        if self.mono and self.n_channels_in == 3:
            # Go from 3 to 1 channel using RGB conversion. This will also
            # work fine if there are just 3 channel grayscale
            X = X[:, :, :, 0] * 0.2125 + X[:, :, :, 1] * 0.7154 + X[:, :, :, 2] * 0.0721

            X = X[:, :, :, np.newaxis]

        if self.mono:
            # Just subtract the mean imagent BGR value, which is as close as we
            # get to vgg19 normalization
            X -= 114.67
        else:
            X = image_utils._preprocess_numpy_input(X)
        X = X.astype("float32", copy=False)
        return X, y
