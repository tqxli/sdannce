from abc import abstractmethod
import imageio, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dannce.engine.skeletons.utils import SYMMETRY, load_body_profile
from dannce.engine.models.vis import draw_voxels
from dannce.engine.data import ops, processing

##################################################################################################
# UTIL_FUNCTIONS
##################################################################################################


def compute_mask_nan_loss(loss_fcn, kpts_gt, kpts_pred):
    # kpts_gt, kpts_pred, notnan = mask_nan(kpts_gt, kpts_pred)
    notnan_gt = ~torch.isnan(kpts_gt)
    notnan = notnan_gt.sum()
    # when ground truth is all NaN for certain reasons, do not compute loss since it results in NaN
    if notnan == 0:
        # print("Found all NaN ground truth")
        return kpts_pred.new_zeros((), requires_grad=True)

    return loss_fcn(kpts_gt[notnan_gt], kpts_pred[notnan_gt]) / notnan


##################################################################################################
# LOSSES
##################################################################################################


class BaseLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def forward(kpts_gt, kpts_pred):
        return NotImplementedError


class L2Loss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        loss = compute_mask_nan_loss(nn.MSELoss(reduction="sum"), kpts_gt, kpts_pred)
        return self.loss_weight * loss


class MSELoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, heatmap_gt, heatmap_pred):
        # if len(heatmap_gt.shape) == 5:
        #     heatmap_gt = heatmap_gt.permute(0, 4, 1, 2, 3)
        loss = F.mse_loss(heatmap_gt, heatmap_pred)

        return self.loss_weight * loss


class BCELoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, heatmap_gt, heatmap_pred):
        if len(heatmap_gt.shape) == 5:
            heatmap_gt = heatmap_gt.permute(0, 4, 1, 2, 3)

        heatmap_pred = torch.sigmoid(heatmap_pred)
        loss = F.binary_cross_entropy(heatmap_pred, heatmap_gt)

        return self.loss_weight * loss


class HuberLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        return F.huber_loss(kpts_gt, kpts_pred)


class BCEWithLogitsLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        return F.binary_cross_entropy_with_logits(kpts_pred, kpts_gt)


class ReconstructionLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, gt, pred):
        loss = F.mse_loss(gt, pred)
        return self.loss_weight * loss


class L1Loss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        loss = compute_mask_nan_loss(nn.L1Loss(reduction="sum"), kpts_gt, kpts_pred)
        return self.loss_weight * loss


class WeightedL1Loss(BaseLoss):
    def __init__(self, joint_weights=None, num_joints=23, **kwargs):
        super().__init__(**kwargs)

        self.weighting = np.ones((num_joints,))
        if isinstance(joint_weights, list):
            for joint, weight in joint_weights:
                self.weighting[joint] = weight

    def forward(self, kpts_gt, kpts_pred):
        loss = []
        for joint_idx in range(kpts_gt.shape[-1]):
            gt, pred = kpts_gt[:, :, joint_idx], kpts_pred[:, :, joint_idx]

            joint_loss = compute_mask_nan_loss(nn.L1Loss(reduction="sum"), gt, pred)
            joint_loss = joint_loss * self.weighting[joint_idx]
            loss.append(joint_loss)

        loss_mean = sum(loss) / len(loss)

        return self.loss_weight * loss_mean


class BoneLengthLoss(BaseLoss):
    def __init__(
        self,
        priors,
        body_profile="rat23",
        mask=None,
        std_multiplier=1,
        upper_only=False,
        relative_scale=False,
        ref_index=3,
        ref_loss_weight=0.1,
        **kwargs,
    ):
        """
        Implements the bone length regularization loss.

        Args:
            priors (str): path to npy file, which holds the pre-computed bone lengths
            body_profile (str): skeleton profile available in `dannce/engine/skeletons`
            mask (List): keypoints to be excluded
            relative_scale (Bool): regularize by relative ratios rather than absolute
            lengths.
            ref_index (int): index of the reference bone, i.e. all bone lengths will be
            rescaled with respect to this specific bone.
            ref_loss_weight (float): besides the relative ratios, also regularize the
            absolute length of the reference bone
        """
        super().__init__(**kwargs)

        self.limbs = torch.LongTensor(
            load_body_profile(body_profile)["limbs"]
        )  # [n_limbs, 2]
        self.priors = np.load(priors, allow_pickle=True)  # [n_limbs, 2]
        self.upper_only = upper_only
        self.std_multiplier = std_multiplier

        # use relative scales instead of actual bone lengths
        self.relative_scale = relative_scale
        if relative_scale:
            self.ref_index = ref_index
            self.ref = self.priors[ref_index, 0]
            self.ref_loss_weight = ref_loss_weight
            self.relative_priors = self.priors / self.ref

        # consider mask out some of the constraints (e.g. Snout-SpineF, 2)
        # to avoid over regularization
        self.mask = mask

        self._construct_intervals(do_masking=(self.mask is not None))

    def _construct_intervals(self, do_masking=False):
        self.intervals = []

        priors = self.relative_priors if self.relative_scale else self.priors
        for idx, (mean, std) in enumerate(priors):
            if (do_masking) and (idx in self.mask):
                self.intervals.append([-10000, 10000])
            elif self.upper_only:
                self.intervals.append([-10000, mean + self.std_multiplier * std])
            else:
                self.intervals.append(
                    [mean - self.std_multiplier * std, mean + self.std_multiplier * std]
                )

        self.intervals = torch.tensor(
            np.stack(self.intervals, axis=0), dtype=torch.float32
        ).unsqueeze(
            0
        )  # [1, n_limbs, 2]
        self.lbound, self.ubound = (
            self.intervals[..., 0],
            self.intervals[..., 1],
        )  # [1, n_limbs]

    def forward(self, kpts_gt, kpts_pred):
        """
        kpts_pred: [bs, 3, n_joints]
        """
        device = kpts_pred.device

        kpts_from = kpts_pred[:, :, self.limbs[:, 0].to(device)]  # [bs, 3, n_limbs]
        kpts_to = kpts_pred[:, :, self.limbs[:, 1].to(device)]  # [bs, 3, n_limbs]
        lens = torch.norm(kpts_from - kpts_to, dim=1, p=2)  # [bs, n_limbs]

        if self.relative_scale:
            loss_ref = (lens[:, self.ref_index] - self.ref).abs().mean()
            lens = lens / lens[:, self.ref_index].unsqueeze(1)

        loss = torch.maximum(
            lens - self.ubound.to(device), torch.zeros(())
        ) + torch.maximum(self.lbound.to(device) - lens, torch.zeros(()))
        loss = loss.mean()

        if self.relative_scale:
            loss += self.ref_loss_weight * loss_ref

        return self.loss_weight * loss


class ConsistencyLoss(BaseLoss):
    def __init__(self, method="l1", copies_per_sample=1, **kwargs):
        """
        Implements the batch consistency loss.

        This loss assumes that the input batch contains augmented copies of
        the same sample(s) and imposes consistency over these copies' final
        predictions.

        Args:
            method (str): distance metric, L1 or L2 Euclidean distance
            copies_per_sample (int): number of unique samples in each batch
        """
        super().__init__(**kwargs)
        assert method in ["l1", "l2"]
        self.method = method
        self.copies_per_sample = copies_per_sample

    def forward(self, kpts_gt, kpts_pred):
        if self.copies_per_sample <= kpts_pred.shape[0]:
            kpts_pred = kpts_pred.reshape(
                -1, self.copies_per_sample, *kpts_pred.shape[1:]
            )
        else:
            # validation
            kpts_pred = kpts_pred.unsqueeze(0)
        diff = torch.diff(kpts_pred, dim=1)
        if self.method == "l1":
            loss_temp = torch.abs(diff).mean()
        else:
            loss_temp = (diff ** 2).sum(1).sqrt().mean()
        return self.loss_weight * loss_temp


class TemporalLoss(BaseLoss):
    def __init__(self, temporal_chunk_size, method="l1", downsample=1, **kwargs):
        super().__init__(**kwargs)

        self.temporal_chunk_size = temporal_chunk_size
        assert method in ["l1", "l2"]
        self.method = method

    def forward(self, kpts_gt, kpts_pred):
        # reshape w.r.t temporal chunk size
        kpts_pred = kpts_pred.reshape(
            -1, self.temporal_chunk_size, *kpts_pred.shape[1:]
        )
        diff = torch.diff(kpts_pred, dim=1)
        if self.method == "l1":
            loss_temp = torch.abs(diff).mean()
        else:
            loss_temp = (diff ** 2).sum(1).sqrt().mean()
        return self.loss_weight * loss_temp


class BoneVectorLoss(BaseLoss):
    def __init__(self, body_profile="rat23", mask=None, **kwargs):
        super().__init__(**kwargs)

        self.animal = body_profile
        self.limbs = torch.LongTensor(
            load_body_profile(body_profile)["limbs"]
        )  # [n_limbs, 2]

        if mask is not None:
            x = torch.arange(len(self.limbs))
            self.limbs = [self.limbs[l] for l in x if l not in mask]

    def get_bone_vectors(self, kpts):
        device = kpts.device
        kpts_from = kpts[:, :, self.limbs[:, 0].to(device)]  # [bs, 3, n_limbs]
        kpts_to = kpts[:, :, self.limbs[:, 1].to(device)]  # [bs, 3, n_limbs]

        return kpts_from - kpts_to

    def forward(self, kpts_gt, kpts_pred):
        bvec_gt = self.get_bone_vectors(kpts_gt)
        bvec_pred = self.get_bone_vectors(kpts_pred)

        loss = torch.norm(bvec_gt - bvec_pred, dim=1).mean()
        return loss * self.loss_weight


class KCSLoss(BaseLoss):
    def __init__(self, body_profile="rat23", **kwargs):
        super().__init__(**kwargs)

        self.animal = load_body_profile(body_profile)
        self.n_joints = 23  # len(self.animal["joints_name"])
        self.limbs = torch.LongTensor(self.animal["limbs"])  # [n_limbs, 2]
        self._construct_c()
        self.l1_loss = L1Loss()

    def _construct_c(self):
        C = torch.zeros(self.n_joints, self.n_joints)
        for col, (r, t) in enumerate(self.limbs):
            C[r, col] = 1
            C[t, col] = -1
        self.C = C

    def forward(self, kpts_gt, kpts_pred):
        # represent the keypoint predictions in the kinematic chain space
        device = kpts_pred.device
        C = self.C.to(device)
        B_pred = torch.mm(kpts_pred, C)
        B_gt = torch.mm(kpts_gt, C)
        kcs_pred = torch.bmm(B_pred.permute(0, 2, 1), B_pred)
        kcs_gt = torch.bmm(B_gt.permute(0, 2, 1), B_gt)

        return self.loss_weight * self.l1_loss(kcs_gt, kcs_pred)


class JointAngleLoss(KCSLoss):
    def __init__(self, angle_prior, std_multiplier=1.0, **kwargs):
        self.priors = np.load(angle_prior, allow_pickle=True)
        super().__init__(**kwargs)

        self.std_multiplier = std_multiplier
        self._construct_intervals()

    def _construct_intervals(self):
        mean, std = self.priors
        self.lbound, self.ubound = (
            mean - self.std_multiplier * std,
            mean + self.std_multiplier,
        )  # [23, 23]
        self.lbound, self.ubound = (
            torch.from_numpy(self.lbound).unsqueeze(0),
            torch.from_numpy(self.ubound).unsqueeze(0),
        )

    def forward(self, kpts_gt, kpts_pred):
        device = kpts_pred.device
        C = self.C.to(device)
        B = kpts_pred @ C
        KCS = B.permute(0, 2, 1) @ B
        angles = (
            KCS
            / torch.norm(B.permute(0, 2, 1), dim=-1, keepdim=True)
            / torch.norm(B, dim=1, keepdim=True)
        )

        diff = torch.maximum(
            angles - self.ubound.to(device), torch.zeros(())
        ) + torch.maximum(
            self.lbound.to(device) - angles, torch.zeros(())
        )  # [B, 23, 23]

        return self.loss_weight * torch.norm(diff, p=1)


class BodySymmetryLoss(BaseLoss):
    def __init__(self, animal="mouse22", **kwargs):
        super().__init__(**kwargs)

        self.animal = animal
        self.limbL = torch.as_tensor([limbs[0] for limbs in SYMMETRY[animal]])
        self.limbR = torch.as_tensor([limbs[1] for limbs in SYMMETRY[animal]])

    def forward(self, kpts_gt, kpts_pred):
        len_L = (
            (torch.diff(kpts_pred[:, :, self.limbL], dim=-1) ** 2).sum(1).sqrt().mean()
        )
        len_R = (
            (torch.diff(kpts_pred[:, :, self.limbR], dim=-1) ** 2).sum(1).sqrt().mean()
        )
        loss = (len_L - len_R).abs()
        return self.loss_weight * loss


class SeparationLoss(BaseLoss):
    def __init__(self, delta=5, **kwargs):
        super().__init__(**kwargs)

        self.delta = delta

    def forward(self, kpts_gt, kpts_pred):
        num_kpts = kpts_pred.shape[-1]

        t1 = kpts_pred.repeat(1, 1, num_kpts)
        t2 = kpts_pred.repeat(1, num_kpts, 1).reshape(t1.shape)

        lensqr = ((t1 - t2) ** 2).sum(1)
        sep = torch.maximum(self.delta - lensqr, 0.0).sum(1) / num_kpts ** 2

        return self.loss_weight * sep.mean()


class GaussianRegLoss(BaseLoss):
    def __init__(self, method="mse", sigma=5, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.sigma = sigma

    def visualize(self, target):
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("TkAgg")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        draw_voxels(target.clone().detach().cpu(), ax)
        plt.show(block=True)
        input("Press Enter to continue...")

    def save(self, heatmaps, heatmap_target, savedir="./debug_gaussian_unsupervised"):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        heatmaps = heatmaps.clone().detach().cpu().numpy()
        heatmap_target = heatmap_target.clone().detach().cpu().numpy()

        for i in range(heatmaps.shape[0]):
            for j in range(heatmaps.shape[1]):
                im = processing.norm_im(heatmaps[i, j]) * 255
                im = im.astype("uint8")
                of = os.path.join(savedir, f"{i}_{j}.tif")
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1]))

        for i in range(heatmap_target.shape[0]):
            for j in range(heatmap_target.shape[1]):
                im = processing.norm_im(heatmap_target[i, j]) * 255
                im = im.astype("uint8")
                of = os.path.join(savedir, f"{i}_{j}_target.tif")
                imageio.mimwrite(of, np.transpose(im, [2, 0, 1]))

    def _generate_gaussian_target(self, centers, grids):
        """
        centers: [batch_size, 3, n_joints]
        grid: [batch_size, 3, h, w, d]
        """
        y_3d = grids.new_zeros(
            centers.shape[0], centers.shape[2], *grids.shape[2:]
        )  # [bs, n_joints, n_vox, n_vox, n_vox]
        for i in range(y_3d.shape[0]):
            for j in range(y_3d.shape[1]):
                y_3d[i, j] = torch.exp(
                    -(
                        (grids[i, 1] - centers[i, 1, j]) ** 2
                        + (grids[i, 0] - centers[i, 0, j]) ** 2
                        + (grids[i, 2] - centers[i, 2, j]) ** 2
                    )
                )
                y_3d[i, j] /= 2 * self.sigma ** 2

        return y_3d

    def forward(self, kpts_gt, kpts_pred, heatmaps, grids):
        """
        kpts_pred: [bs, 3, n_joints]
        heatmaps: [bs, n_joints, h, w, d]
        grid_centers: [bs, h*w*d, 3]
        """
        # reshape grids
        grids = grids.permute(0, 2, 1).reshape(
            grids.shape[0], grids.shape[2], *heatmaps.shape[2:]
        )  # [bs, 3, n_vox, n_vox, n_vox]

        if grids.shape[0] != kpts_pred.shape[0]:
            grids = torch.stack((grids, grids), dim=1)  # [bs, 2, n_vox, n_vox, n_vox]
            grids = grids.reshape(-1, *grids.shape[2:])

        # generate gaussian shaped targets based on current predictions
        gaussian_gt = self._generate_gaussian_target(
            kpts_pred, grids
        )  # [bs, n_joints, n_vox**3]
        heatmaps = heatmaps.sigmoid()

        # compute loss
        if self.method == "mse":
            loss = 0.5 * F.mse_loss(gaussian_gt, heatmaps)
        elif self.method == "cross_entropy":
            loss = F.binary_cross_entropy(heatmaps, gaussian_gt)

        return self.loss_weight * loss


class PairConsistencyLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_gt, kpts_pred):
        """
        Assume each batch contains social volumes
        kpts_gt, kpts_pred: [B*2, 3, n_joints]
        """
        kpts_gt = kpts_gt.reshape(-1, 2, *kpts_gt.shape[1:])
        kpts_pred = kpts_pred.reshape(
            -1, 2, *kpts_pred.shape[1:]
        )  # [B, 2, 3, n_joints]

        kpts_gt = kpts_gt.reshape(-1, 2, *kpts_gt.shape[2:])  # [b, 2, 2, 3, n_joints]
        kpts_pred = kpts_pred.reshape(-1, 2, *kpts_pred.shape[2:])

        gt_notnan = ~torch.isnan(kpts_gt[:, 0])
        gt_counter_notnan = ~torch.isnan(kpts_gt[:, 1].flip(1))
        notnan = gt_notnan & gt_counter_notnan

        if notnan.sum() == 0:
            return kpts_pred.new_zeros((), requires_grad=True)

        animal1_pred = kpts_pred[:, 0]  # [B//2, 2, 3, n_joints]
        animal2_pred = kpts_pred[:, 1].flip(1)  # also in order of [animal1, animal2]

        loss = (
            F.l1_loss(animal1_pred[notnan], animal2_pred[notnan], reduction="sum")
            / notnan.sum()
        )

        return loss * self.loss_weight


class PairRepulsionLoss(BaseLoss):
    def __init__(self, delta=5, pairwise=False, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

        # whether only penalize repulsion on the same set of keypoints (e.g. elbow of animal 1 and elbow of animal 2)
        self.pairwise = pairwise

    def forward(self, kpts_gt, kpts_pred):
        """
        [bs, 3, n_joints]
        """
        bs, n_joints = kpts_pred.shape[0], kpts_pred.shape[-1]

        kpts_pred = kpts_pred.reshape(
            -1, 2, *kpts_pred.shape[1:]
        )  # [n_pairs, 2, 3, n_joints]

        if self.pairwise:
            diff = torch.diff(kpts_pred, axis=1).squeeze()  # [n_pairs, 3, n_joints]
            dist = (diff ** 2).sum(1).sqrt()  # [n_pairs, n_joints]

        else:
            # compute the distance between each joint of animal 1 and all joints of animal 2
            a1 = kpts_pred[:, 0, :, :].repeat(
                1, 1, n_joints
            )  # [n_pairs, 3, n_joints^2]
            a2 = kpts_pred[:, 1, :, :].repeat(1, n_joints, 1).reshape(a1.shape)

            diffsqr = (a1 - a2) ** 2
            dist = diffsqr.sum(1).sqrt()  # [n_pairs, n_joints^2]

        # only penalize distance <= specified threshold delta
        dist = torch.maximum(self.delta - dist, torch.zeros_like(dist))

        return self.loss_weight * dist.mean()


class SilhouetteLoss(BaseLoss):
    def __init__(self, delta=5, reduction_axes=[2, 3, 4], **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.reduction_axes = reduction_axes

    def forward(self, vh, heatmaps):
        """
        vh, heatmaps: [bs, n_joints, H, W, D]
        Heatmap is not softmaxed.
        """
        prob = ops.spatial_softmax(heatmaps)

        sil = torch.sum(vh * prob, axis=self.reduction_axes)
        sil = torch.mean(-(sil + 1e-12).log())
        if torch.isnan(sil):
            sil = sil.new_zeros((), requires_grad=True)

        return self.loss_weight * sil


class SilhouetteLoss2D(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, sils, heatmaps):
        """
        vh: [bs, n_cams, H, W, D]
        heatmaps: [bs, n_joints, H, W, D]. Heatmap is not softmaxed.
        """
        prob = ops.spatial_softmax(heatmaps).unsqueeze(1)  # [bs, 1, n_joints, H, W, D]
        sils = sils.unsqueeze(2)  # [bs, n_cams, 1, H, W, D]
        sils = torch.sum(sils * prob, axis=[3, 4, 5])  # [bs, n_cams, n_joints, H, W, D]
        sils = torch.mean(-(sils + 1e-12).log())
        if torch.isnan(sils):
            sils = sils.new_zeros((), requires_grad=True)

        return self.loss_weight * sils


class VarianceLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, kpts_pred, heatmaps, grids):
        """
        heatmaps: [bs, n_joints, h, w, d]
        grid: [bs, h*w*d, 3]
        kpts_pred: [bs, 3, n_joints]
        """
        prob = ops.spatial_softmax(heatmaps)

        gridsize = prob.shape[2:]
        grids = grids.reshape(grids.shape[0], 1, *gridsize, 3)  # [bs, 1, h, w, d, 3]
        kpts_pred = (
            kpts_pred.permute(0, 2, 1).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        )  # [bs, n_joints, 1, 1, 1, 3]

        diff = torch.sum((grids - kpts_pred).sqrt(), dim=-1)  # [bs, n_joints, h, w, d]
        diff *= prob

        loss = self.loss_weight * torch.mean(torch.sum(diff, dim=[2, 3, 4]))

        return loss if not torch.isnan(loss) else loss.new_zeros((), requires_grad=True)
