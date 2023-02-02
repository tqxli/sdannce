"""Operations for dannce."""
import numpy as np
import cv2
import time
from typing import Text
import torch
import torch.nn.functional as F


class Camera:

    def __init__(self, R, t, K, tdist, rdist, name=""):
        self.R = torch.tensor(R).float()  # rotation matrix
        assert self.R.shape == (3, 3)

        self.t = torch.tensor(t).float()  # translation vector
        assert self.t.shape == (1, 3)

        self.K = torch.tensor(K).float()  # intrinsic matrix
        assert self.K.shape == (3, 3)

        self.extrinsics = torch.cat((self.R, self.t), dim=0)  # extrinsics
        self.M = self.extrinsics @ self.K  # camera matrix

        # distortion
        self.tdist = torch.tensor(tdist).squeeze().float()
        self.rdist = torch.tensor(rdist).squeeze().float()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[2, 0], self.K[2, 1]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[2, 0], self.K[2, 1] = new_cx, new_cy
        self.M = self.extrinsics @ self.K

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[2, 0], self.K[2, 1]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[2, 0], self.K[2, 1] = new_fx, new_fy, new_cx, new_cy
        self.M = self.extrinsics @ self.K

    def camera_matrix(self):
        return self.M

    def extrinsics(self):
        return self.extrinsics


def camera_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Derive the camera matrix.

    Derive the camera matrix from the camera intrinsic matrix (K),
    and the extrinsic rotation matric (R), and extrinsic
    translation vector (t).

    Note that this uses the matlab convention, such that
    M = [R;t] * K
    """
    return np.concatenate((R, t), axis=0) @ K


def world_to_cam(pts, M, device):
    M = M.to(device=device)
    pts1 = torch.ones(pts.shape[0], 1, dtype=torch.float32, device=device)

    projPts = torch.matmul(torch.cat((pts, pts1), 1), M)
    return projPts


def project_to2d(pts, M: np.ndarray, device: Text) -> torch.Tensor:
    """Project 3d points to 2d.

    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """

    # pts = torch.Tensor(pts.copy()).to(device)
    if isinstance(M, torch.Tensor):
        M = M.to(device=device)
    else:
        M = torch.as_tensor(M, device=device)
    pts1 = torch.ones(pts.shape[0], 1, dtype=torch.float32, device=device)

    projPts = torch.matmul(torch.cat((pts, pts1), 1), M)
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:]

    return projPts


def sample_grid_nearest(im: np.ndarray, projPts: np.ndarray,
                        device: Text) -> torch.Tensor:
    """Unproject features."""
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,
    feats = torch.as_tensor(im.copy(),
                            device=device) if not torch.is_tensor(im) else im
    grid = projPts
    c = int(round(projPts.shape[0]**(1 / 3.0)))

    fh, fw, fdim = list(feats.shape)

    # # make sure all projected indices fit onto the feature map
    im_x = torch.clamp(grid[:, 0], 0, fw - 1)
    im_y = torch.clamp(grid[:, 1], 0, fh - 1)

    im_xr = im_x.round().type(torch.long)
    im_yr = im_y.round().type(torch.long)
    im_xr[im_xr < 0] = 0
    im_yr[im_yr < 0] = 0
    Ir = feats[im_yr, im_xr]

    return Ir.reshape((c, c, c, -1)).permute(3, 0, 1, 2).unsqueeze(0)


def sample_grid_linear(im: np.ndarray, projPts: np.ndarray,
                       device: Text) -> torch.Tensor:
    """Unproject features."""
    # im_x, im_y are the x and y coordinates of each projected 3D position.
    # These are concatenated here for every image in each batch,

    feats = torch.as_tensor(im.copy(),
                            device=device) if not torch.is_tensor(im) else im
    grid = projPts
    c = int(round(projPts.shape[0]**(1 / 3.0)))

    fh, fw, fdim = list(feats.shape)

    # # make sure all projected indices fit onto the feature map
    im_x = torch.clamp(grid[:, 0], 0, fw - 1)
    im_y = torch.clamp(grid[:, 1], 0, fh - 1)

    # round all indices
    im_x0 = torch.floor(im_x).type(torch.long)
    # new array with rounded projected indices + 1
    im_x1 = im_x0 + 1
    im_y0 = torch.floor(im_y).type(torch.long)
    im_y1 = im_y0 + 1

    # Convert from int to float -- but these are still round
    # numbers because of rounding step above
    im_x0_f, im_x1_f = im_x0.type(torch.float), im_x1.type(torch.float)
    im_y0_f, im_y1_f = im_y0.type(torch.float), im_y1.type(torch.float)

    # Gather  values
    # Samples all featuremaps at the projected indices,
    # and their +1 counterparts. Stop at Ia for nearest neighbor interpolation.

    # need to clip the corner indices because they might be out of bounds...
    # This could lead to different behavior compared to TF/numpy, which return 0
    # when an index is out of bounds
    im_x1_safe = torch.clamp(im_x1, 0, fw - 1)
    im_y1_safe = torch.clamp(im_y1, 0, fh - 1)

    im_x1[im_x1 < 0] = 0
    im_y1[im_y1 < 0] = 0
    im_x0[im_x0 < 0] = 0
    im_y0[im_y0 < 0] = 0
    im_x1_safe[im_x1_safe < 0] = 0
    im_y1_safe[im_y1_safe < 0] = 0

    Ia = feats[im_y0, im_x0]
    Ib = feats[im_y0, im_x1_safe]
    Ic = feats[im_y1_safe, im_x0]
    Id = feats[im_y1_safe, im_x1_safe]

    # To recaptiulate behavior  in numpy/TF, zero out values that fall outside bounds
    Ib[im_x1 > fw - 1] = 0
    Ic[im_y1 > fh - 1] = 0
    Id[(im_x1 > fw - 1) | (im_y1 > fh - 1)] = 0
    # Calculate bilinear weights
    # We've now sampled the feature maps at corners around the projected values
    # Here, the corners are weighted by distance from the projected value
    wa = (im_x1_f - im_x) * (im_y1_f - im_y)
    wb = (im_x1_f - im_x) * (im_y - im_y0_f)
    wc = (im_x - im_x0_f) * (im_y1_f - im_y)
    wd = (im_x - im_x0_f) * (im_y - im_y0_f)

    Ibilin = (wa.unsqueeze(1) * Ia + wb.unsqueeze(1) * Ib +
              wc.unsqueeze(1) * Ic + wd.unsqueeze(1) * Id)

    return Ibilin.reshape((c, c, c, -1)).permute(3, 0, 1, 2).unsqueeze(0)


def sample_grid(im: np.ndarray,
                projPts: np.ndarray,
                device: Text,
                method: Text = "linear"):
    """Transfer 3d features to 2d by projecting down to 2d grid, using torch.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    if method == "nearest" or method == "out2d":
        proj_rgb = sample_grid_nearest(im, projPts, device)
    elif method == "linear" or method == "bilinear":
        proj_rgb = sample_grid_linear(im, projPts, device)
    else:
        raise Exception("{} not a valid interpolation method".format(method))

    return proj_rgb


def unDistortPoints(
    pts,
    intrinsicMatrix,
    radialDistortion,
    tangentDistortion,
    rotationMatrix,
    translationVector,
):
    """Remove lens distortion from the input points.

    Input is size (M,2), where M is the number of points
    """
    dcoef = radialDistortion.ravel()[:2].tolist() + tangentDistortion.ravel().tolist()

    if len(radialDistortion.ravel()) == 3:
        dcoef = dcoef + [radialDistortion.ravel()[-1]]
    else:
        dcoef = dcoef + [0]

    ts = time.time()
    pts_u = cv2.undistortPoints(
        np.reshape(pts, (-1, 1, 2)).astype("float32"),
        intrinsicMatrix.T,
        np.array(dcoef),
        P=intrinsicMatrix.T,
    )

    pts_u = np.reshape(pts_u, (-1, 2))

    return pts_u


def triangulate(pts1, pts2, cam1, cam2):
    """Return triangulated 3- coordinates.

    Following Matlab convetion, given lists of matching points, and their
    respective camera matrices, returns the triangulated 3- coordinates.
    pts1 and pts2 must be Mx2, where M is the number of points with
    (x,y) positions. M 3-D points will be returned after triangulation
    """
    pts1 = pts1.T
    pts2 = pts2.T

    cam1 = cam1.T
    cam2 = cam2.T

    out_3d = np.zeros((3, pts1.shape[1]))

    for i in range(out_3d.shape[1]):
        if ~np.isnan(pts1[0, i]):
            pt1 = pts1[:, i:i + 1]
            pt2 = pts2[:, i:i + 1]

            A = np.zeros((4, 4))
            A[0:2, :] = pt1 @ cam1[2:3, :] - cam1[0:2, :]
            A[2:, :] = pt2 @ cam2[2:3, :] - cam2[0:2, :]

            u, s, vh = np.linalg.svd(A)
            v = vh.T

            X = v[:, -1]
            X = X / X[-1]

            out_3d[:, i] = X[0:3].T
        else:
            out_3d[:, i] = np.nan

    return out_3d


def triangulate_multi_instance(pts, cams):
    """Return triangulated 3- coordinates.

    Following Matlab convetion, given lists of matching points, and their
    respective camera matrices, returns the triangulated 3- coordinates.
    pts1 and pts2 must be Mx2, where M is the number of points with
    (x,y) positions. M 3-D points will be returned after triangulation
    """

    pts = [pt.T for pt in pts]
    cams = [c.T for c in cams]
    out_3d = np.zeros((3, pts[0].shape[1]))
    # traces = np.zeros((out_3d.shape[1],))

    for i in range(out_3d.shape[1]):
        if ~np.isnan(pts[0][0, i]):
            p = [p[:, i:i + 1] for p in pts]

            A = np.zeros((2 * len(cams), 4))
            for j in range(len(cams)):
                A[(j) * 2:(j + 1) *
                  2] = p[j] @ cams[j][2:3, :] - cams[j][0:2, :]

            u, s, vh = np.linalg.svd(A)
            v = vh.T

            X = v[:, -1]
            X = X / X[-1]

            out_3d[:, i] = X[0:3].T
            # traces[i] = np.sum(s[0:3])

        else:
            out_3d[:, i] = np.nan

    return out_3d


def ravel_multi_index(I, J, shape):
    """Create an array of flat indices from coordinate arrays.

    shape is (rows, cols)
    """
    r, c = shape
    return I * c + J


def distortPoints(points, intrinsicMatrix, radialDistortion,
                  tangentialDistortion, device):
    """Distort points according to camera parameters.
    Ported from Matlab 2018a
    """

    # unpack the intrinsic matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = torch.as_tensor((cx, cy), dtype=torch.float32, device=device)
    centeredPoints = points - center

    # normalize the pcenteredPoints[:, 1] / fyoints
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm**2 + yNorm**2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3, ))
    k[:2] = radialDistortion[:2]
    if list(radialDistortion.shape)[0] < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm**2)
    dyTangential = p[0] * (r2 + 2 * yNorm**2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = torch.transpose(torch.stack((xNorm, yNorm)), 0, 1)

    distortedNormalizedPoints = (
        normalizedPoints +
        normalizedPoints * torch.transpose(torch.stack((alpha, alpha)), 0, 1) +
        torch.transpose(torch.stack((dxTangential, dyTangential)), 0, 1))

    distortedPointsX = (distortedNormalizedPoints[:, 0] * fx + cx +
                        skew * distortedNormalizedPoints[:, 1])

    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy

    distortedPoints = torch.stack((distortedPointsX, distortedPointsY))

    return distortedPoints


def cal_reprojection_error(keypoints_3d,
                           keypoints_2d,
                           joint_idx,
                           cameras,
                           camnames,
                           prefix=None):
    reprojection_errs = []
    keypoints_3d = torch.as_tensor(keypoints_3d[np.newaxis, :])
    for cam in camnames:
        camname = cam if prefix is None else prefix + cam
        pts = keypoints_2d[cam]["COM"][joint_idx]
        pts = pts[np.newaxis, :]

        camparam = cameras[camname]
        proj = project_to2d(keypoints_3d, camparam["cammat"], "cpu")[:, :2]
        proj = proj.cpu().numpy()
        proj = unDistortPoints(
            proj,
            camparam["K"],
            camparam["RDistort"],
            camparam["TDistort"],
            camparam["R"],
            camparam["t"],
        )

        err = np.mean(np.sqrt(np.sum((proj - pts)**2, axis=1)))
        reprojection_errs.append(err)

    reprojection_errs = np.vstack(reprojection_errs).T

    return reprojection_errs[0]


def expected_value_3d(prob_map, grid_centers):
    bs, channels, h, w, d = prob_map.shape

    prob_map = prob_map.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    grid_centers = grid_centers.reshape(-1, 3)
    weighted_centers = prob_map.unsqueeze(1) * grid_centers.unsqueeze(-1)
    weighted_centers = weighted_centers.reshape(-1, h * w * d, 3,
                                                channels).sum(1)

    return weighted_centers  # [bs, 3, channels]


def max_coord_3d(heatmaps):
    heatmaps = spatial_softmax(heatmaps)
    bs, channels, h, w, d = heatmaps.shape

    accu_x = heatmaps.sum(dim=3)
    accu_x = accu_x.sum(dim=3)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=2)
    accu_z = accu_z.sum(dim=2)

    accu_x = accu_x * torch.arange(h).float().to(heatmaps.device)
    accu_y = accu_y * torch.arange(w).float().to(heatmaps.device)
    accu_z = accu_z * torch.arange(d).float().to(heatmaps.device)

    x = accu_x.sum(dim=2, keepdim=True)
    y = accu_y.sum(dim=2, keepdim=True)
    z = accu_z.sum(dim=2, keepdim=True)

    # normalize to [-1, 1] for subsequent grid sampling
    x = x / float(h) - 0.5
    y = y / float(w) - 0.5
    z = z / float(d) - 0.5
    preds = torch.cat((z, y, x), dim=2)
    preds *= 2

    return preds


def expected_value_2d(prob_map, grid):
    bs, channels, h, w = prob_map.shape

    prob_map = prob_map.permute(0,
                                2, 3, 1).reshape(bs, -1, channels).unsqueeze(
                                    2)  #[bs, h*w, 1, channels]
    weighted_centers = prob_map * grid  #[bs, h*w, 2, channels]

    return weighted_centers.sum(1)  #[bs, 2, channels]


def spatial_softmax(feats):
    """
    can work with 2D or 3D
    """
    bs, channels = feats.shape[:2]
    feat_shape = feats.shape[2:]
    feats = feats.reshape(bs, channels, -1)
    feats = F.softmax(feats, dim=-1)
    return feats.reshape(bs, channels, *feat_shape)


def var_3d(prob_map, grid_centers, markerlocs):
    """Return the average variance across all marker probability maps.

    Used a loss to promote "peakiness" in the probability map output
    prob_map should be (batch_size,h,w,d,channels)
    grid_centers should be (batch_size,h*w*d,3)
    markerlocs is (batch_size,3,channels)
    """
    channels, h, w, d = prob_map.shape[1:]
    prob_map = prob_map.permute(0, 2, 3, 3, 1).reshape(-1, channels)
    grid_dist = (grid_centers.unsqueeze(-1) - markerlocs.unsqueeze(1))**2
    grid_dist = grid_dist.sum(2)
    grid_dist = grid_dist.reshape(-1, channels)

    weighted_var = prob_map * grid_dist
    weighted_var = weighted_var.reshape(-1, h * w * d, channels)
    weighted_var = weighted_var.sum(1)
    return torch.mean(weighted_var, dim=-1, keepdim=True)
