import numpy as np


def mask_to_bbox(mask):
    bounding_boxes = np.zeros((4,))
    y, x, _ = np.where(mask != 0)
    try:
        bounding_boxes[0] = np.min(x)
        bounding_boxes[1] = np.min(y)
        bounding_boxes[2] = np.max(x)
        bounding_boxes[3] = np.max(y)
    except:
        return bounding_boxes
    return bounding_boxes


def mask_iou(mask1, mask2):
    """ compute iou between two binary masks
    """
    intersection = np.sum(mask1 * mask2)
    if intersection == 0:
        return 0.0
    union = np.sum(np.logical_or(mask1, mask2).astype(np.uint8))
    return intersection / union


def mask_intersection(mask1, mask2):
    return mask1 * mask2


def bbox_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[0] - bb1[2]) * (bb1[1] - bb1[3])
    bb2_area = (bb2[0] - bb2[2]) * (bb2[1] - bb2[3])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def compute_support(coms, mask, support_region_size=10):
    counts = []
    for i in range(len(coms)):
        index = coms[i]  # .clone().cpu().int().numpy()
        sp_l = np.maximum(0, index[1] - support_region_size)
        sp_r = np.minimum(mask.shape[0], index[1] + support_region_size)
        sp_t = np.maximum(0, index[0] - support_region_size)
        sp_b = np.minimum(mask.shape[1], index[0] + support_region_size)

        count = np.sum(mask[int(sp_l) : int(sp_r), int(sp_t) : int(sp_b), 0])
        counts.append(count)
    return np.array(counts)


def extract_3d_sil(vol):
    """
    vol: [n_samples, H, W, D, C*n_cam]
    """
    vol[vol > 0] = 1
    vol = np.sum(vol, axis=-1, keepdims=True)

    # TODO: want max over each sample, instead of all
    upper_thres = np.max(vol)

    vol[vol < upper_thres] = 0
    vol[vol > 0] = 1

    print(
        "{}\% of silhouette training voxels are occupied".format(
            100 * np.sum(vol) / len(vol.ravel())
        )
    )
    return vol


def extract_3d_sil_soft(vol, keeprange=3):
    vol[vol > 0] = 1
    vol = np.sum(vol, axis=-1, keepdims=True)

    upper_thres = np.max(vol)
    lower_thres = upper_thres - keeprange
    vol[vol <= lower_thres] = 0
    vol[vol > 0] = (vol[vol > 0] - lower_thres) / keeprange

    print(
        "{}\% of silhouette training voxels are occupied".format(
            100 * np.sum((vol > 0)) / len(vol.ravel())
        )
    )
    return vol


def compute_bbox_from_3dmask(mask3d, grids):
    """
    mask3d: [N, H, W, D, 1]
    grid: [N, H*W*D, 3]
    """
    new_com3ds, new_dims = [], []
    for mask, grid in zip(mask3d, grids):
        mask = np.squeeze(mask)  # [H, W, D]
        h, w, d = np.where(mask)

        h_l, h_u = h.min(), h.max()
        w_l, w_u = w.min(), w.max()
        d_l, d_u = d.min(), d.max()

        corner1 = np.array([h_l, w_l, d_l])
        corner2 = np.array([h_u, w_u, d_u])
        mid_point = ((corner1 + corner2) / 2).astype(int)

        grid = np.reshape(grid, (*mask.shape, 3))

        new_com3d = grid[mid_point[0], mid_point[1], mid_point[2]]

        new_dim = (
            grid[corner2[0], corner2[1], corner2[2]]
            - grid[corner1[0], corner1[1], corner1[2]]
        )

        new_com3ds.append(new_com3d)
        new_dims.append(new_dim)

    new_com3ds = np.stack(new_com3ds, axis=0)
    new_dims = np.stack(new_dims, axis=0)

    return new_com3ds, new_dims
