import os
import random
import cv2
import numpy as np
from dannce.engine.data import processing
import warnings
import scipy.io as sio

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

MISSING_KEYPOINTS_MSG = (
    "If mirror augmentation is used, the right_keypoints indices and left_keypoints "
    + "indices must be specified as well. "
    + "For the skeleton, ['RHand, 'LHand', 'RFoot', 'LFoot'], "
    + "set right_keypoints: [0, 2] and left_keypoints: [1, 3] in the config file"
)

class PoseDatasetFromMem(torch.utils.data.Dataset):
    """Generate 3d conv data from memory.

    Attributes:
        augment_brightness (bool): If True, applies brightness augmentation
        augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
        augment_hue (bool): If True, applies hue augmentation
        bright_val (float): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
        chan_num (int): Number of input channels
        data (np.ndarray): Image volumes
        expval (bool): If True, crafts input for an AVG network
        hue_val (float): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
        indexes (np.ndarray): Sample indices used for batch generation
        labels (Dict): Label dictionary
        list_IDs (List): List of sampleIDs
        nvox (int): Number of voxels in each grid dimension
        random (bool): If True, shuffles camera order for each batch
        rotation (bool): If True, applies rotation augmentation in 90 degree increments
        rotation_val (float): Range of angles used for continuous rotation augmentation
        shuffle (bool): If True, shuffle the samples before each epoch
        var_reg (bool): If True, returns input used for variance regularization
        xgrid (np.ndarray): For the AVG network, this contains the 3D grid coordinates
        n_rand_views (int): Number of reviews to sample randomly from the full set
        replace (bool): If True, samples n_rand_views with replacement
        aux_labels (np.ndarray): If not None, contains the 3D MAX training targets for AVG+MAX training.
        temporal_chunk_list (np.ndarray, optional): If not None, contains chunked sampleIDs -- useful when loading in temporally contiguous samples
    """

    def __init__(
        self,
        list_IDs,
        data,
        labels,
        rotation=True,
        random=True,
        chan_num=3,
        shuffle=True,
        expval=False,
        xgrid=None,
        var_reg=False,
        nvox=64,
        augment_brightness=True,
        augment_hue=True,
        augment_continuous_rotation=True,
        mirror_augmentation=False,
        right_keypoints=None,
        left_keypoints=None,
        bright_val=0.05,
        hue_val=0.05,
        rotation_val=5,
        replace=True,
        n_rand_views=None,
        heatmap_reg=False,
        heatmap_reg_coeff=0.01,
        aux_labels=None,
        temporal_chunk_list=None,
        occlusion=False,
        pairs=None,
        transformed_batch=False,
    ):
        """Initialize data generator.
        """
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.rotation = rotation
        self.random = random
        self.chan_num = chan_num
        self.shuffle = shuffle
        self.expval = expval
        self.augment_hue = augment_hue
        self.augment_continuous_rotation = augment_continuous_rotation
        self.augment_brightness = augment_brightness
        self.mirror_augmentation = mirror_augmentation
        self.right_keypoints = right_keypoints
        self.left_keypoints = left_keypoints
        if self.mirror_augmentation and (
            self.right_keypoints is None or self.left_keypoints is None
        ):
            raise Exception(MISSING_KEYPOINTS_MSG)
        self.var_reg = var_reg
        self.xgrid = xgrid
        self.nvox = nvox
        self.bright_val = bright_val
        self.hue_val = hue_val
        self.rotation_val = rotation_val
        self.n_rand_views = n_rand_views
        self.replace = replace
        self.heatmap_reg = heatmap_reg
        self.heatmap_reg_coeff = heatmap_reg_coeff
        self.aux_labels = aux_labels
        self.temporal_chunk_list = temporal_chunk_list

        self.occlusion = occlusion
        # if self.occlusion:
        #     assert aux_labels is not None, "Missing aux labels for occlusion training."
        
        self.pairs = pairs
        if self.pairs is not None:
            self.temporal_chunk_size = len(self.pairs[0])
        
        self._update_temporal_batch_size()

    def __len__(self):
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        if self.temporal_chunk_list is not None:
           return len(self.temporal_chunk_list)
        
        if self.pairs is not None:
            return len(self.pairs)

        return len(self.list_IDs)
    
    def _update_temporal_batch_size(self):
        self.temporal_chunk_size = 1
        if self.temporal_chunk_list is not None:
            self.temporal_chunk_size = len(self.temporal_chunk_list[0])

        if self.pairs is not None:
            self.temporal_chunk_size = len(self.pairs[0])

    def __getitem__(self, index):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        if self.temporal_chunk_list is not None:
            list_IDs_temp = self.temporal_chunk_list[index]
        elif self.pairs is not None:
            list_IDs_temp = index
        else:
            list_IDs_temp = [self.list_IDs[index]]
        X, X_grid, y_3d, aux = self.__data_generation(list_IDs_temp)
        return X, X_grid, y_3d, aux

    def rot90(self, X):
        """Rotate X by 90 degrees CCW.

        Args:
            X (np.ndarray): Image volume or grid

        Returns:
            X (np.ndarray): Rotated image volume or grid
        """
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    def mirror(self, X, y_3d, X_grid):
        # Flip the image and x coordinates about the x axis
        X = X[:, ::-1, ...]
        X_grid = X_grid[:, ::-1, ...]

        # Flip the left and right keypoints.
        temp = y_3d[..., self.left_keypoints].copy()
        y_3d[..., self.left_keypoints] = y_3d[..., self.right_keypoints]
        y_3d[..., self.right_keypoints] = temp
        return X, y_3d, X_grid

    def rot180(self, X):
        """Rotate X by 180 degrees.

        Args:
            X (np.ndarray): Image volume or grid

        Returns:
            X (np.ndarray): Rotated image volume or grid
        """
        X = X[::-1, ::-1, :, :]
        return X

    def random_rotate(self, X, y_3d, aux=None):
        """Rotate each sample by 0, 90, 180, or 270 degrees.

        Args:
            X (np.ndarray): Image volumes
            y_3d (np.ndarray): 3D grid coordinates (AVG) or training target volumes (MAX)
            aux (np.ndarray or None): Populated in MAX+AVG mode with the training target volumes
        Returns:
            X (np.ndarray): Rotated image volumes
            y_3d (np.ndarray): Rotated 3D grid coordinates (AVG) or training target volumes (MAX)
        """
        # rotation for all volumes within each chunk must be consistent
        rot = np.random.choice(np.arange(4), 1)
        
        if rot == 0:
            pass
        elif rot == 1:
            # Rotate180
            for j in range(self.temporal_chunk_size):
                X[j], y_3d[j] = self.rot180(X[j]), self.rot180(y_3d[j])
                if aux is not None:
                    aux[j] = self.rot180(aux[j])
        elif rot  == 2:
            # Rotate90
            for j in range(self.temporal_chunk_size):
                X[j], y_3d[j] = self.rot90(X[j]), self.rot90(y_3d[j])
                if aux is not None:
                    aux[j] = self.rot90(aux[j])
        elif rot == 3:
            # Rotate -90/270
            for j in range(self.temporal_chunk_size):
                X[j], y_3d[j] = self.rot180(self.rot90(X[j])), self.rot180(self.rot90(y_3d[j]))
                if aux is not None:
                    aux[j] = self.rot180(self.rot90(aux[j]))
    
        if aux is not None:
            return X, y_3d, aux
        return X, y_3d
    
    def random_continuous_rotation(self, X, y_3d, max_delta=5):
        """Rotates X and y_3d a random amount around z-axis.

        Args:
            X (np.ndarray): input image volume
            y_3d (np.ndarray): 3d target (for MAX network) or voxel center grid (for AVG network)
            max_delta (int, optional): maximum range for rotation angle.

        Returns:
            np.ndarray: rotated image volumes
            np.ndarray: rotated grid coordimates
        """
        # torchvision.transforms.functional.affine - input: [..., H, W]
        rotangle = np.random.rand() * (2 * max_delta) - max_delta
        X = torch.from_numpy(X).reshape(*X.shape[:3], -1).permute(0, 3, 1, 2) # dimension [B, D*C, H, W]
        y_3d = torch.from_numpy(y_3d).reshape(y_3d.shape[:3], -1).permute(0, 3, 1, 2)
        for i in range(X.shape[0]):
            X[i] = TF.affine(X[i], angle=rotangle)
            y_3d[i] = TF.affine(y_3d[i], angle=rotangle)

        X = X.permute(0, 2, 3, 1).reshape(*X.shape[:3], X.shape[2], -1).numpy()
        y_3d = y_3d.permute(0, 2, 3, 1).reshape(*X.shape[:3], X.shape[2], -1).numpy()

        return X, y_3d

    def visualize(self, original, augmented):
        """Plots example image after augmentation

        Args:
            original (np.ndarray): image before augmentation
            augmented (np.ndarray): image after augmentation.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Original image")
        plt.imshow(original)

        plt.subplot(1, 2, 2)
        plt.title("Augmented image")
        plt.imshow(augmented)
        plt.show()
        input("Press Enter to continue...")

    def do_augmentation(self, X, X_grid, y_3d, aux=None):
        """Applies augmentation

        Args:
            X (np.ndarray): image volumes
            X_grid (np.ndarray): 3D grid coordinates
            y_3d (np.ndarray): training targets
            aux (np.ndarray or None): additional target volumes if using MAX+AVG mode

        Returns:
            X (np.ndarray): Augemented image volumes
            X_grid (np.ndarray): 3D grid coordinates
            y_3d (np.ndarray): Training targets
        """
        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, 3),
                )
                if aux is not None:
                    X, X_grid, aux = self.random_rotate(X.copy(), X_grid.copy(), aux.copy())
                else:
                    X, X_grid = self.random_rotate(X.copy(), X_grid.copy())
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.temporal_chunk_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X.copy(), y_3d.copy())

        if self.augment_continuous_rotation and aux is None:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, 3),
                )
                X, X_grid = self.random_continuous_rotation(
                    X.copy(), X_grid.copy(), self.rotation_val
                )
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.temporal_chunk_size, -1, 3))
            else:
                X, y_3d = self.random_continuous_rotation(
                    X.copy(), y_3d.copy(), self.rotation_val
                )

        if self.augment_hue and self.chan_num == 3:
            # […, 1 or 3, H, W]
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X_temp = torch.from_numpy(X[..., channel_ids]).permute(0, 4, 1, 2, 3) #[bs, 3, H, W, D]
                X_temp = X_temp.reshape(*X_temp.shape[:3], -1) #[bs, 3, H, W*D]
                random_hue_val = float(torch.empty(1).uniform_(-self.hue_val, self.hue_val))
                X_temp = TF.adjust_hue(X_temp, random_hue_val)
                X[..., channel_ids] = X_temp.reshape(*X_temp.shape[:3], X_temp.shape[2], -1).permute(0, 2, 3, 4, 1).numpy()

        elif self.augment_hue:
            warnings.warn(
                "Trying to augment hue with an image that is not RGB. Skipping."
            )

        if self.augment_brightness:
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X_temp = torch.as_tensor(X[..., channel_ids]).permute(0, 4, 1, 2, 3)
                X_temp = X_temp.reshape(*X_temp.shape[:3], -1) #[bs, 3, H, W*D]
                random_bright_val = float(torch.empty(1).uniform_(1-self.bright_val, 1+self.bright_val))
                X_temp = TF.adjust_brightness(X_temp, random_bright_val)
                X[..., channel_ids] = X_temp.reshape(*X_temp.shape[:3], X_temp.shape[2], -1).permute(0, 2, 3, 4, 1).numpy()

        if self.mirror_augmentation and self.expval and aux is None:
            if np.random.rand() > 0.5:
                X_grid = np.reshape(
                    X_grid,
                    (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, 3),
                )
                # Flip the image and the symmetric keypoints
                X, y_3d, X_grid = self.mirror(X.copy(), y_3d.copy(), X_grid.copy())
                X_grid = np.reshape(X_grid, (self.temporal_chunk_size, -1, 3))
        else:
            pass
            ##TODO: implement mirror augmentation for max and avg+max modes

        # if self.occlusion:
        #     if np.random.rand() > 0.5:
        #         occlusion_idx = np.random.choice(self.__len__())
        #         rand_cam = np.random.choice(int(X.shape[-1] // self.chan_num)-1)
        #         foreground_obj = self.aux_labels[occlusion_idx:(occlusion_idx+1), :, :, :, rand_cam*3:(rand_cam+1)*3]
        #         occluded_area = (foreground_obj != -1)
        #         X[..., rand_cam*3:(rand_cam+1)*3][occluded_area] = foreground_obj[occluded_area]
                
        return X, X_grid, y_3d, aux

    def do_random(self, X):
        """Randomly re-order camera views

        Args:
            X (np.ndarray): image volumes

        Returns:
            X (np.ndarray): Shuffled image volumes
        """

        if self.random:
            X = X.reshape((*X.shape[:4], self.chan_num, -1), order="F")
            X = X[..., np.random.permutation(X.shape[-1])]
            X = X.reshape((*X.shape[:4], -1), order="F")

        if self.n_rand_views is not None:
            # Select a set of cameras randomly with replacement.
            X = X.reshape((*X.shape[:4], self.chan_num, -1), order="F")
            if self.replace:
                X = X[..., np.random.randint(X.shape[-1], size=(self.n_rand_views,))]
            else:
                if not self.random:
                    raise Exception(
                        "For replace=False for n_rand_views, random must be turned on"
                    ) 
                X = X[..., :self.n_rand_views]
            X = X.reshape((*X.shape[:4], -1), order="F")

        return X

    def get_max_gt_ind(self, X_grid, y_3d):
        """Uses the gt label position to find the index of the voxel corresponding to it.
        Used for heatmap regularization.
        """

        diff = np.sum(
            (X_grid[:, :, :, np.newaxis] - y_3d[:, np.newaxis, :, :]) ** 2, axis=2
        )
        inds = np.argmin(diff, axis=1)
        grid_d = int(np.round(X_grid.shape[1] ** (1 / 3)))
        inds = np.unravel_index(inds, (grid_d, grid_d, grid_d))
        return np.stack(inds, axis=1)
    
    def _convert_numpy_to_tensor(self, X, X_grid, y_3d, aux):
        if X_grid is not None:
            X_grid = torch.from_numpy(X_grid)
        if aux is not None:
            aux = torch.from_numpy(aux).permute(0, 4, 1, 2, 3)

        return torch.from_numpy(X).permute(0, 4, 1, 2, 3), X_grid, torch.from_numpy(y_3d), aux

    def __data_generation(self, list_IDs_temp):
        """
        X : (chunk_size, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Chunked training samples
                X: Input volumes
                y_3d: Targets
        Raises:
            Exception: For replace=False for n_rand_views, random must be turned on.
        """
        if self.pairs is None:
            # Initialization        
            X = np.zeros((self.temporal_chunk_size, *self.data.shape[1:]))
            y_3d = np.zeros((self.temporal_chunk_size, *self.labels.shape[1:]))

            # Only used for AVG mode
            if self.expval:
                X_grid = np.zeros((self.temporal_chunk_size, *self.xgrid.shape[1:]))
            else:
                X_grid = None

            # Only used for AVG+MAX mode
            if (not self.occlusion) and (self.aux_labels is not None):
                aux = np.zeros((self.temporal_chunk_size, *self.aux_labels.shape[1:]))
            else:
                aux = None

            for i, ID in enumerate(list_IDs_temp):
                X[i] = self.data[ID].copy()
                y_3d[i] = self.labels[ID]
                if self.expval:
                    X_grid[i] = self.xgrid[ID]
                if aux is not None:
                    aux[i] = self.aux_labels[ID]
        else:
            ID = list_IDs_temp
            X = self.data[ID].copy()
            y_3d = self.labels[ID]

            if self.expval:
                X_grid = self.xgrid[ID]
            else:
                X_grid = None

            if self.aux_labels is not None:
                aux = self.aux_labels[ID]
            else:
                aux = None

        X, X_grid, y_3d, aux = self.do_augmentation(X, X_grid, y_3d, aux)
        # Randomly re-order, if desired
        X = self.do_random(X)
        
        return self._convert_numpy_to_tensor(X, X_grid, y_3d, aux)
    
    def compute_avg_bone_length(self):
        return


class PoseDatasetNPY(PoseDatasetFromMem):
    """Generates 3d conv data from npy files.

    Attributes:
    augment_brightness (bool): If True, applies brightness augmentation
    augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
    augment_hue (bool): If True, applies hue augmentation
    batch_size (int): Batch size
    bright_val (float): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
    chan_num (int): Number of input channels
    labels_3d (Dict): training targets
    expval (bool): If True, crafts input for an AVG network
    hue_val (float): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
    indexes (np.ndarray): Sample indices used for batch generation
    list_IDs (List): List of sampleIDs
    nvox (int): Number of voxels in each grid dimension
    random (bool): If True, shuffles camera order for each batch
    rotation (bool): If True, applies rotation augmentation in 90 degree increments
    rotation_val (float): Range of angles used for continuous rotation augmentation
    shuffle (bool): If True, shuffle the samples before each epoch
    var_reg (bool): If True, returns input used for variance regularization
    n_rand_views (int): Number of reviews to sample randomly from the full set
    replace (bool): If True, samples n_rand_views with replacement
    imdir (Text): Name of image volume npy subfolder
    griddir (Text): Name of grid volumw npy subfolder
    mono (bool): If True, return monochrome image volumes
    sigma (float): For MAX network, size of target Gaussian (mm)
    cam1 (bool): If True, prepares input for training a single camea network
    prefeat (bool): If True, prepares input for a network performing volume feature extraction before fusion
    npydir (Dict): path to each npy volume folder for each recording (i.e. experiment)
    """

    def __init__(
        self,
        list_IDs,
        labels_3d,
        npydir,
        # batch_size,
        imdir="image_volumes",
        griddir="grid_volumes",
        aux=False,
        auxdir="visual_hulls",
        prefeat=False,        
        mono=False,
        cam1=False,    
        sigma=10,
        **kwargs
    ):
        """Generates 3d conv data from npy files.

        Args:
            list_IDs (List): List of sampleIDs
            labels_3d (Dict): training targets
            npydir (Dict): path to each npy volume folder for each recording (i.e. experiment)
            batch_size (int): Batch size
            imdir (Text, optional): Name of image volume npy subfolder
            griddir (Text, optional): Name of grid volumw npy subfolder
            mono (bool, optional): If True, return monochrome image volumes
            cam1 (bool, optional): If True, prepares input for training a single camea network
            prefeat (bool, optional): If True, prepares input for a network performing volume feature extraction before fusion
            sigma (float, optional): For MAX network, size of target Gaussian (mm)
        """
        super(PoseDatasetNPY, self).__init__(
            list_IDs=list_IDs,
            data=None,
            labels=None,
            **kwargs
        )
        self.labels_3d = labels_3d
        self.npydir = npydir
        self.griddir = griddir
        self.imdir = imdir
        self.mono = mono
        self.cam1 = cam1
        #self.replace = replace
        self.prefeat = prefeat
        self.sigma = sigma
        self.auxdir = auxdir
        self.aux = aux

    def __getitem__(self, index):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        if self.temporal_chunk_list is not None:
            indices = self.temporal_chunk_list[index]
            list_IDs_temp = [self.list_IDs[idx] for idx in indices]
        elif self.pairs is not None:
            list_IDs_temp = self.pairs[index]
        else:
            list_IDs_temp = [self.list_IDs[index]]
        # Generate data
        X, X_grid, y_3d, aux = self.__data_generation(list_IDs_temp)
        return X, X_grid, y_3d, aux

    def _downscale_occluded_views(self, X, occlusion_scores):
        """
        X: [H, W, D, n_cams*3]
        occlusion_scores: [n_cams]
        """
        # occluded_X = np.reshape(X.copy(), (*X.shape[:3], occlusion_scores.shape[-1], -1))
        # occlusion_scores = occlusion_scores[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

        # occluded_X *= (1-occlusion_scores) # the occlusion score is determined by IoU

        # occluded_X = np.reshape(occluded_X, (*X.shape[:3], -1))
        occluded_views = (occlusion_scores > 0.7)

        occluded = np.where(occluded_views)[0]
        unoccluded = np.where(~occluded_views)[0]
        
        if len(occluded) == 0:
            return X
        
        X = np.reshape(X, (*X.shape[:3], -1, 3)) #[H, W, D, n_cam, C]
        
        alternatives = np.random.choice(unoccluded, len(occluded), replace=(len(unoccluded) <= len(occluded)))
        X[:, :, :, occluded, :] = X[:, :, :, alternatives, :]
        # print(f"Replace view {occluded} with {alternatives}")

        X = np.reshape(X, (*X.shape[:3], -1)) 

        return X
    
    def _save_3d_targets(self, listIDs, y_3d, savedir='debug_MAX_target'):
        import imageio
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for i in range(y_3d.shape[0]):
            for j in range(y_3d.shape[-1]):
                im = processing.norm_im(y_3d[i, :, :, :, j]) * 255
                im = im.astype("uint8")
                of = os.path.join(savedir, f"{listIDs[i]}_{j}.tif")
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1]))
    
    def _save_3d_inputs(self, listIDs, X, savedir='debug_MAX_input'):
        import imageio
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for i in range(X.shape[0]):
            for j in range(6):
                im = X[
                    i,
                    :,
                    :,
                    :,
                    j * 3 : (j + 1) * 3,
                ]
                im = processing.norm_im(im) * 255
                im = im.astype("uint8")
                of = os.path.join(
                    savedir,
                    listIDs[i] + "_cam" + str(j) + ".tif",
                )
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1, 3]))

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d or y_3d_max: Targets
        Raises:
            Exception: For replace=False for n_rand_views, random must be turned on.
        """

        # Initialization

        X = []
        y_3d = []
        X_grid = []
        aux = []

        for i, ID in enumerate(list_IDs_temp):
            # Need to look up the experiment ID to get the correct directory
            IDkey = ID.split("_")
            eID = int(IDkey[0])
            sID = IDkey[1]

            vol = np.load(
                os.path.join(self.npydir[eID], self.imdir, "0_" + sID + ".npy")
                ).astype("float32")

            if self.occlusion:
                occlusion_scores = np.load(os.path.join(self.npydir[eID], 'occlusion_scores', "0_" + sID + ".npy")).astype("float32")
                vol = self._downscale_occluded_views(vol, occlusion_scores)
            X.append(vol)

            y_3d.append(self.labels_3d[ID])
            X_grid.append(
                np.load(
                    os.path.join(self.npydir[eID], self.griddir, "0_" + sID + ".npy")
                )
            )

            if self.aux:
                a = np.load(os.path.join(self.npydir[eID], self.auxdir, "0_" + sID + ".npy")).astype("float32")
                aux.append(a)

        X = np.stack(X)
        y_3d = np.stack(y_3d)

        X_grid = np.stack(X_grid)
        aux = np.stack(aux) if len(aux) != 0 else None

        if not self.expval:
            y_3d_max = np.zeros(
                (self.temporal_chunk_size, self.nvox, self.nvox, self.nvox, y_3d.shape[-1])
            )

        if not self.expval:
            X_grid = np.reshape(X_grid, (-1, self.nvox, self.nvox, self.nvox, 3))
            for gridi in range(X_grid.shape[0]):
                x_coord_3d = X_grid[gridi, :, :, :, 0]
                y_coord_3d = X_grid[gridi, :, :, :, 1]
                z_coord_3d = X_grid[gridi, :, :, :, 2]
                for j in range(y_3d_max.shape[-1]):
                    y_3d_max[gridi, :, :, :, j] = np.exp(
                        -(
                            (y_coord_3d - y_3d[gridi, 1, j]) ** 2
                            + (x_coord_3d - y_3d[gridi, 0, j]) ** 2
                            + (z_coord_3d - y_3d[gridi, 2, j]) ** 2
                        )
                        / (2 * self.sigma ** 2)
                    )
            X_grid = np.reshape(X_grid, (X_grid.shape[0], -1, 3))
            y_3d = y_3d_max

        ncam = int(X.shape[-1] // self.chan_num)
        # breakpoint()
        X, X_grid, y_3d, aux = self.do_augmentation(X, X_grid, y_3d, aux)

        # Randomly re-order, if desired
        X = self.do_random(X)
        
        if self.mono and self.chan_num == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = np.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    self.chan_num,
                    -1,
                ),
                order="F",
            )
            X = (
                X[:, :, :, :, 0] * 0.2125
                + X[:, :, :, :, 1] * 0.7154
                + X[:, :, :, :, 2] * 0.0721
            )

        if self.cam1:
            # collapse the cameras to the batch dimensions.
            X = np.reshape(
                X,
                (X.shape[0], X.shape[1], X.shape[2], X.shape[3], self.chan_num, -1),
                order="F",
            )
            X = np.transpose(X, [0, 5, 1, 2, 3, 4])
            X = np.reshape(X, (-1, X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
            if self.expval:
                y_3d = np.tile(y_3d, [ncam, 1, 1])
                X_grid = np.tile(X_grid, [ncam, 1, 1])
            else:
                y_3d = np.tile(y_3d, [ncam, 1, 1, 1, 1])

        X = processing.preprocess_3d(X) 

        return self._convert_numpy_to_tensor(X, X_grid, y_3d, aux)


class COMDatasetFromMem(torch.utils.data.Dataset):
    def __init__(
        self,
        list_IDs,
        data,
        labels,
        batch_size,
        chan_num=3,
        shuffle=True,
        augment_brightness=False,
        augment_hue=False,
        augment_rotation=False,
        augment_zoom=False,
        augment_shear=False,
        augment_shift=False,
        bright_val=0.05,
        hue_val=0.05,
        shift_val=0.05,
        rotation_val=5,
        shear_val=5,
        zoom_val=0.05,
    ):
        """Initialize data generator."""
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.chan_num = chan_num
        self.shuffle = shuffle

        self.augment_brightness = augment_brightness
        self.augment_hue = augment_hue
        self.augment_rotation = augment_rotation
        self.augment_zoom = augment_zoom
        self.augment_shear = augment_shear
        self.augment_shift = augment_shift
        self.bright_val = bright_val
        self.hue_val = hue_val
        self.shift_val = shift_val
        self.rotation_val = rotation_val
        self.shear_val = shear_val
        self.zoom_val = zoom_val   

    def __len__(self):
        return len(self.list_IDs)

    def shift_im(self, im, lim, dim=2):
        ulim = im.shape[dim] - np.abs(lim)

        if dim == 2:
            if lim < 0:
                im[:, :, :ulim] = im[:, :, np.abs(lim) :]
                im[:, :, ulim:] = im[:, :, ulim : ulim + 1]
            else:
                im[:, :, lim:] = im[:, :, :ulim]
                im[:, :, :lim] = im[:, :, lim : lim + 1]
        elif dim == 1:
            if lim < 0:
                im[:, :ulim] = im[:, np.abs(lim) :]
                im[:, ulim:] = im[:, ulim : ulim + 1]
            else:
                im[:, lim:] = im[:, :ulim]
                im[:, :lim] = im[:, lim : lim + 1]
        else:
            raise Exception("Not a valid dimension for shift indexing")

        return im

    def random_shift(self, X, y_2d, im_h, im_w, scale):
        """
        Randomly shifts all images in batch, in the range [-im_w*scale, im_w*scale]
            and [im_h*scale, im_h*scale]
        """
        wrng = np.random.randint(-int(im_w * scale), int(im_w * scale))
        hrng = np.random.randint(-int(im_h * scale), int(im_h * scale))

        X = self.shift_im(X, wrng)
        X = self.shift_im(X, hrng, dim=1)

        y_2d = self.shift_im(y_2d, wrng)
        y_2d = self.shift_im(y_2d, hrng, dim=1)

        return X, y_2d

    def __getitem__(self, index):
        list_IDs_temp = [self.list_IDs[index]]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples."""
        # Initialization

        X = np.zeros((self.batch_size, *self.data.shape[1:]))
        y_2d = np.zeros((self.batch_size, *self.labels.shape[1:]))

        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.data[ID].copy()
            y_2d[i] = self.labels[ID]

        if self.augment_rotation or self.augment_shear or self.augment_zoom:

            affine = {}
            affine["zoom"] = 1
            affine["rotation"] = 0
            affine["shear"] = 0

            # Because we use views down below,
            # don't change the targets in memory.
            # But also, don't deep copy y_2d unless necessary (that's
            # why it's here and not above)
            y_2d = y_2d.copy()

            # TODO: replace with torchvision.transforms
            if self.augment_rotation:
                affine["rotation"] = self.rotation_val * (np.random.rand() * 2 - 1)
            # if self.augment_zoom:
            #     affine["zoom"] = self.zoom_val * (np.random.rand() * 2 - 1) + 1
            if self.augment_shear:
                affine["shear"] = self.shear_val * (np.random.rand() * 2 - 1)

            X = TF.affine(
                torch.from_numpy(X).permute(0, 3, 1, 2),
                angle=affine["rotation"],
                shear=affine["shear"],
            )
            y_2d = TF.affine(
                torch.from_numpy(y_2d).permute(0, 3, 1, 2),
                angle=affine["rotation"],
                shear=affine["shear"],
            )

        if self.augment_shift:
            X, y_2d = self.random_shift(
                X, y_2d.copy(), X.shape[1], X.shape[2], self.shift_val
            )

        if self.augment_brightness:
            X_temp = torch.as_tensor(X).permute(0, 3, 1, 2) #[bs, 3, H, W]
            random_bright_val = float(torch.empty(1).uniform_(1-self.bright_val, 1+self.bright_val))
            X_temp = TF.adjust_brightness(X_temp, random_bright_val)
            X = X_temp.permute(0, 2, 3, 1).numpy() 
        
        if self.augment_hue:
            if self.chan_num == 3:
                X_temp = torch.as_tensor(X).permute(0, 3, 1, 2) #[bs, 3, H, W]
                random_hue_val = float(torch.empty(1).uniform_(-self.hue_val, self.hue_val))
                X_temp = TF.adjust_hue(X_temp, random_hue_val)
                X = X_temp.permute(0, 2, 3, 1).numpy() 
            else:
                warnings.warn("Hue augmention set to True for mono. Ignoring.")

        X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
        y_2d = torch.from_numpy(y_2d).permute(0, 3, 1, 2).float()
        return X, y_2d


class MultiViewImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images,
        grids,
        labels_3d,
        cameras,
    ):
        super(MultiViewImageDataset, self).__init__()
        self.images = images
        self.grids = grids
        self.labels_3d = labels_3d
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        X = processing.preprocess_3d(self.images[idx])
        X_grid = self.grids[idx]
        y_3d = self.labels_3d[idx]
        camera = self.cameras[idx]

        return X, X_grid, camera, y_3d


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        num_joints, 
        imdir=None, labeldir=None, images=None, labels=None, 
        grids=None, labels_3d=None, cameras=None,
        return_Gaussian=True, 
        sigma=2,
        image_size=[256, 256],
        heatmap_size=[64, 64],
        train=True
    ):
        super(ImageDataset, self).__init__()
        self.images = images
        self.labels = labels

        self.read_frommem = (self.images is not None)

        if not self.read_frommem:
            self.imdir = imdir 
            self.labeldir = labeldir

            self.imlist = sorted(os.listdir(imdir))
            self.annot = sorted(os.listdir(labeldir))
            assert len(self.imlist) == len(self.annot)

        self.num_joints = num_joints
        self.return_Gaussian = return_Gaussian
        self.sigma = sigma

        self.image_size = np.array(image_size)
        self.heatmap_size = np.array(heatmap_size)
        self.ds_fac = image_size[0] // heatmap_size[0]

        self.grids = grids
        self.labels_3d = labels_3d
        self.return_3d = (grids is not None) and (labels_3d is not None)

        self.cameras = cameras
        self.return_cameras = (cameras is not None)

        self.train = train
        self._transforms()

    def __len__(self):
        if self.read_frommem:
            return self.images.shape[0]
        return len(self.imlist)
    
    def _vis_heatmap(self, im, target):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121)
        ax.imshow(im.permute(1, 2, 0).numpy().astype(np.uint8))

        # ax.scatter(kpts2d[0].numpy(), kpts2d[1].numpy())

        ax = fig.add_subplot(122)
        ax.imshow(target.sum(0).numpy())

        plt.show(block=True)
        input("Press Enter to continue...")
    
    def _generate_Gaussian_target(self, labels):
        (x_coord, y_coord) = np.meshgrid(
            np.arange(self.heatmap_size[1]), np.arange(self.heatmap_size[0])
        )
        
        targets = []
        for joint in range(labels.shape[-1]):
            if np.isnan(labels[:, joint]).sum() == 0:
                target = np.exp(
                        -(
                            (y_coord - labels[1, joint] // self.ds_fac) ** 2
                            + (x_coord - labels[0, joint] // self.ds_fac) ** 2
                        )
                        / (2 * self.sigma ** 2)
                    )
            else:
                target = np.zeros((self.heatmap_size[1], self.heatmap_size[0]))
            targets.append(target)
        # crop out and keep the max to be 1 might still work...
        targets = np.stack(targets, axis=0)
        targets = torch.from_numpy(targets).float()

        return targets

        # target_weight = joints.new_ones((self.num_joints, 1))
        # target = joints.new_zeros(self.num_joints, self.heatmap_size[1], self.heatmap_size[0])

        # tmp_size = self.sigma * 3

        # for joint_id in range(self.num_joints):
        #     feat_stride = self.image_size / self.heatmap_size

        #     mu_x = int(joints[0, joint_id] / feat_stride[0] + 0.5)
        #     mu_y = int(joints[1, joint_id] / feat_stride[1] + 0.5)
        #     # Check that any part of the gaussian is in-bounds
        #     ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        #     br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        #     if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
        #             or br[0] < 0 or br[1] < 0:
        #         # If not, just return the image as is
        #         target_weight[joint_id] = 0
        #         continue

        #     # # Generate gaussian
        #     size = 2 * tmp_size + 1
        #     x = torch.arange(0, size, 1)
        #     y = x.unsqueeze(-1)
        #     x0 = y0 = size // 2
        #     # The gaussian is not normalized, we want the center value to equal 1
        #     g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

        #     # Usable gaussian range
        #     g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
        #     g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
        #     # Image range
        #     img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
        #     img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

        #     v = target_weight[joint_id]
        #     if v > 0.5:
        #         target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
        #             g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        # return target, target_weight

    def _transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            # transforms.ToTensor(),
            normalize,
        ])

    def __getitem__(self, idx):
        if self.read_frommem:
            im = self.images[idx].clone()
            targets = self.labels[idx]
        else:
            im = cv2.imread(
                os.path.join(self.imdir, self.imlist[idx]),  
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            annot = np.load(os.path.join(self.labeldir, self.annot[idx]), allow_pickle=True)[()]
            x1, y1, x2, y2 = annot["bbox"]
            w, h = x2-x1, y2-y1
            max_side = max(w, h)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            x1 = int(center[0]-max_side/2)
            x2 = int(center[0]+max_side/2)
            y1 = int(center[1]-max_side/2)
            y2 = int(center[1]+max_side/2)

            kpts2d = annot["keypoints"]

            im = im[int(y1):int(y2), int(x1):int(x2), :]
            kpts2d[0, :] -= int(x1)
            kpts2d[1, :] -= int(y1)

            ori_size = im.shape[:2]
            im = cv2.resize(im, tuple(self.image_size))
            kpts2d[0, :] *= (im.shape[1] / ori_size[1])
            kpts2d[1, :] *= (im.shape[0] / ori_size[0])

        im = im # self.transforms(im).float()
        # kpts2d = torch.from_numpy(kpts2d)

        if self.return_Gaussian:
            if not self.return_3d:
                targets = self._generate_Gaussian_target(targets.numpy())
            else:
                # start = time.time()
                all_targets = []
                for i in range(targets.shape[0]):
                    temp = self._generate_Gaussian_target(targets[i].numpy())
                    all_targets.append(temp)
                targets = np.stack(all_targets, axis=0)
                targets = torch.from_numpy(targets).float() 
                # end = time.time()
                # print("Create gaussian takes {} seconds".format(end-start))
            if self.train:
                if random.random() > 0.5:
                    im = TF.hflip(im)
                    targets = TF.hflip(targets)

                # Random vertical flipping
                if random.random() > 0.5:
                    im = TF.vflip(im)
                    targets = TF.vflip(targets)

                # Random rotation
                if random.random() > 0.5:
                    rot = random.randint(0, 3) * 90
                    if rot != 0:
                        im = TF.rotate(im, rot)
                        targets = TF.rotate(targets, rot)
        # breakpoint()
        # self._vis_heatmap(im[0], targets[0])

        if self.return_3d:
            grid = self.grids[idx]
            y3d = self.labels_3d[idx]
            if self.return_cameras:
                cam = self.cameras[idx]
                return im, targets, cam, grid, y3d
            return im, targets, grid, y3d

        return im, targets


class RAT7MSeqDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root='/media/mynewdrive/datasets/rat7m/annotations',
        seqlen = 16,
        downsample=4,
        keep_joints = [10, 14, 8, 9, 17, 16, 12, 13, 3, 5, 4],
    ):
        super().__init__()

        self.root = root
        self.mocaps = sorted(os.listdir(root))
        self.downsample = downsample
        self.seqlen = seqlen
        # compensate the differences between RAT7M and manually labeled rat data
        self.keep_joints = np.array(keep_joints)

        self.data = self._load_all_mocaps()

        self._chunking()
        self._keep_overlap_joints()
        self._filter_nan()

    def _load_all_mocaps(self):
        return [self.load_mocap(os.path.join(self.root, m))[::self.downsample] for m in self.mocaps]
    
    def _chunking(self):
        for i, data in enumerate(self.data):
            n_chunks = data.shape[0] // self.seqlen
            self.data[i] = np.reshape(data[:n_chunks*self.seqlen], (n_chunks, self.seqlen, *data.shape[-2:]))
        
        self.data = np.concatenate(self.data, axis=0) #[N_CHUNKS, SEQLEN, 3, N_JOINTS]

    def _keep_overlap_joints(self):
        self.data = self.data[..., self.keep_joints]
    
    def _filter_nan(self):
        notnan = ~np.isnan(np.reshape(self.data, (self.data.shape[0], -1)))
        notnan = np.all(notnan, axis=-1)
        self.data = self.data[notnan, ...]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        return sample.reshape(sample.shape[0], -1)
    
    @property
    def input_seqlen(self):
        return self.data.shape[1]
    
    @property
    def input_shape(self):
        return self.data.shape[2]*self.data.shape[3]

    @classmethod
    def load_mocap(self, path):
        d = sio.loadmat(path, struct_as_record=False)
        dataset = vars(d["mocap"][0][0])

        markernames = dataset['_fieldnames']

        mocap = []
        for i in range(len(markernames)):
            mocap.append(dataset[markernames[i]])

        return np.stack(mocap, axis=2) #[N_FRAMES, 3, N_JOINTS]
    
    @classmethod
    def vis_seq(self, seq, savepath, vidname):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter

        CONNECTIVITY = [
            # (0, 1), (0, 2), (1, 2), (2, 3), (3, 6), (5, 7), (6, 7), (17, 18), (16, 19), (10, 11), (14, 15),
            # (3, 4), (3, 12), (3, 13), (4, 5), (5, 8), (5, 9), (8, 17), (9, 16), (12, 10), (13, 14), 
            (8, 10), (8, 6), (8, 7), (9, 10), (10, 2), (10, 3), (2, 4), (3, 5), (0, 6), (1, 7),
        ]
        metadata = dict(title='rat7m', artist='Matplotlib')
        writer = FFMpegWriter(fps=2, metadata=metadata)

        # set up save path
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        # xliml, xlimr = seq[:, 0, :].min(), seq[:, 0, :].max()
        # yliml, ylimr = seq[:, 1, :].min(), seq[:, 1, :].max()
        # zliml, zlimr = seq[:, 2, :].min(), seq[:, 2, :].max()

        with writer.saving(fig, os.path.join(savepath, f'{vidname}.mp4'), dpi=300):
            for i in range(seq.shape[0]):
                pose = seq[i]
                ax.scatter3D(pose[0], pose[1], pose[2], color='k')

                for (index_from, index_to) in CONNECTIVITY:
                    xs, ys, zs = [np.array([pose[k, index_from], pose[k, index_to]]) for k in range(3)]
                    ax.plot3D(xs, ys, zs, c='dodgerblue', lw=2)
                
                # ax.set_xlim(-100, 150)
                # ax.set_ylim(100, 250)
                # ax.set_zlim(0, 150)
                    
                writer.grab_frame()
                ax.clear()
        plt.close()


class RAT7MImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="/media/mynewdrive/datasets/rat7m",
        annot="final_annotations_w_correct_clusterIDs.pkl",
        imgdir='images_unpacked',
        train=True,
        downsample=1,
    ):
        super().__init__()

        self.root = root
        self.train = train
        experiments_train = ['s1-d1', 's2-d1', 's2-d2', 's3-d1', 's4-d1']
        experiments_test = ['s5-d1', 's5-d2']
        self.experiments = experiments_train + experiments_test

        # load annotations from disk
        self.annot_dict = annot_dict = np.load(os.path.join(root, annot), allow_pickle=True)

        # select subjects
        if train:
            filter = np.where(annot_dict["table"]["subject_idx"] != 5)[0][::downsample]
        else:
            filter = np.where(annot_dict["table"]["subject_idx"] == 5)[0][::downsample]

        labels, coms, impaths = [], [], []
        for camname in annot_dict["camera_names"]:
            labels.append(annot_dict["table"]["2D_keypoints"][camname][filter])
            coms.append(annot_dict["table"]["2D_com"][camname][filter])
            impaths.append(annot_dict["table"]["image_paths"][camname][filter])
        self.labels = np.concatenate(labels)
        self.coms = np.concatenate(coms)
        self.impaths = np.concatenate(impaths)
        self.n_samples = len(self.labels)
        
        self._prepare_cameras()
        self._transforms()

        self.dim_crop = [512, 512]
        self.dim_out = [256, 256]

        self.out_scale = 2
        self.rotation_val = 15
        self.ds_fac = self.dim_crop[0] / self.dim_out[0]
        
    def __len__(self):
        return self.n_samples
    
    def _prepare_cameras(self):
        cameras = {}
        camnames = self.annot_dict["camera_names"]
        for i, expname in enumerate(self.experiments):
            subject_idx, day_idx = expname.split('-')
            subject_idx, day_idx = int(subject_idx[-1]), int(day_idx[-1])
        
            cameras[i] = {}
            for camname in camnames:  
                new_params = {}
                old_params = self.annot_dict["cameras"][subject_idx][day_idx][camname]
                new_params["K"] = old_params["IntrinsicMatrix"]
                new_params["R"] = old_params["rotationMatrix"]
                new_params["t"] = old_params["translationVector"]
                new_params["RDistort"] = old_params["RadialDistortion"]
                new_params["TDistort"] = old_params["TangentialDistortion"]
                cameras[i][str(i)+"_"+camname] = new_params

        self.cameras = cameras

    def _crop_im(self, im, com, labels):
        im, cropdim = processing.cropcom(im, com, size=self.dim_crop[0]) 
        labels[0, :] -= cropdim[2]
        labels[1, :] -= cropdim[0]
    
        return im, labels
    
    def _transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def _vis_heatmap(self, im, target, kpts2d):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121)
        ax.imshow(im.permute(1, 2, 0).numpy().astype(np.uint8))

        ax.scatter(kpts2d[0], kpts2d[1])

        ax = fig.add_subplot(122)
        ax.imshow(target.sum(0).numpy())

        plt.show(block=True)
        input("Press Enter to continue...")

    def __getitem__(self, idx):
        impath = os.path.join(self.root, self.impaths[idx])
        im = cv2.imread(impath)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        labels = np.transpose(self.labels[idx], (1, 0))
        com = self.coms[idx]
        im, labels = self._crop_im(im, com, labels)

        im = processing.downsample_batch(im[np.newaxis, ...], fac=self.ds_fac)
        im = np.squeeze(im)
        labels /= self.ds_fac

        # generate gaussian targets
        (x_coord, y_coord) = np.meshgrid(
            np.arange(self.dim_out[1] // 4), np.arange(self.dim_out[0] // 4)
        )
        
        targets = []
        for joint in range(labels.shape[-1]):
            # target = np.zeros((self.dim_out[0], self.dim_out[1]))
            target = np.exp(
                    -(
                        (y_coord - labels[1, joint] // 4) ** 2
                        + (x_coord - labels[0, joint] // 4) ** 2
                    )
                    / (2 * self.out_scale ** 2)
                )
            # tmp_size = 3*self.out_scale

            #target[-tmp_size+int(labels[1, joint]):tmp_size+int(labels[1, joint]), -tmp_size+int(labels[0, joint]):tmp_size+int(labels[0, joint])] *= 10
            # = g[]
            targets.append(target)
        # crop out and keep the max to be 1 might still work...
        targets = np.stack(targets, axis=0)

        # im = np.transpose(im, (2, 0, 1))
        im = self.transforms(im).float()
        targets = torch.from_numpy(targets).float()

        # Random horizontal flipping
        if self.train:
            if random.random() > 0.5:
                im = TF.hflip(im)
                targets = TF.hflip(targets)

            # Random vertical flipping
            if random.random() > 0.5:
                im = TF.vflip(im)
                targets = TF.vflip(targets)

            # Random rotation
            if random.random() > 0.5:
                rot = random.randint(0, 3) * 90
                if rot != 0:
                    im = TF.rotate(im, rot)
                    targets = TF.rotate(targets, rot)
        # im = processing._preprocess_numpy_input(im)
        #im, targets = torch.from_numpy(im).permute(2, 0, 1).float(), 

        # apply transformations
        #rot = self.rotation_val * (np.random.rand() * 2 - 1) if self.train else 0
        #im = TF.affine(im, angle=rot, translate=[0, 0], scale=1.0, shear=0)
        #target = TF.affine(im, angle=rot, translate=[0, 0], scale=1.0, shear=0)

        return im, targets


class RAT7MNPYDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root="/media/mynewdrive/datasets/rat7m",
        annot="final_annotations_w_correct_clusterIDs.pkl",
        train=False,
    ):
        super().__init__()

        self.root = root
        self.exps = ["s5-d1", "s5-d2"] if not train else ["s1-d1", "s2-d1", "s2-d2", "s3-d1", "s4-d1"]
        # load annotations from disk
        self.annot_dict = annot_dict = np.load(os.path.join(root, annot), allow_pickle=True)["table"]
        
        vol_paths, grid_paths = [], []
        for e in self.exps:
            sid, did = int(e[1]), int(e[-1])
            filter = (annot_dict["subject_idx"] == sid) & (annot_dict["day_idx"] == did)
            frames = annot_dict["frame_idx"]["Camera1"][filter]
            # frames = sorted(frames, key=lambda x:int(x))
            fnames = [f"0_{frame}.npy" for frame in frames]
            vp = [os.path.join(root, "npy_volumes", e, "image_volumes", f) for f in fnames]
            gp = [os.path.join(root, "npy_volumes", e, "grid_volumes", f) for f in fnames]
            vol_paths += vp 
            grid_paths += gp

        self.vol_paths = vol_paths
        self.grid_paths = grid_paths

        # just check for sure
        for vp in self.vol_paths:
            assert os.path.exists(vp)
        for gp in self.grid_paths:
            assert os.path.exists(gp)        

        assert len(self.vol_paths) == len(self.grid_paths)
    
    def __len__(self):
        return len(self.vol_paths)
    
    def __getitem__(self, idx):
        X = np.load(self.vol_paths[idx], allow_pickle=True).astype("float32")
        X_grid = np.load(self.grid_paths[idx], allow_pickle=True)

        X = processing.preprocess_3d(X)
        X = X[np.newaxis, :, :, :, :]
        X_grid = X_grid[np.newaxis, :, :]

        return [X, X_grid], [None]

