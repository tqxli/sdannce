import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import maximum_filter
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean as dsm


def _preprocess_numpy_input(x, data_format="channels_last", mode="torch"):
    """Preprocesses a Numpy array encoding a batch of images.
    Args:
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
        - caffe: will convert the images from RGB to BGR,
            then will zero-center each color channel with
            respect to the ImageNet dataset,
            without scaling.
        - tf: will scale pixels between -1 and 1,
            sample-wise.
        - torch: will scale pixels between 0 and 1 and then
            will normalize each channel with respect to the
            ImageNet dataset.
    Returns:
        Preprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype("float32", copy=False)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
            mean = [103.939, 116.779, 123.68]
            std = None

    # Zero-center by mean pixel
    if data_format == "channels_first":
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
            else:
                x[:, 0, :, :] -= mean[0]
                x[:, 1, :, :] -= mean[1]
                x[:, 2, :, :] -= mean[2]
        if std is not None:
            x[:, 0, :, :] /= std[0]
            x[:, 1, :, :] /= std[1]
            x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


def downsample_batch(imstack, fac=2, method="PIL"):
    """Downsample each image in a batch."""

    if method == "PIL":
        out = np.zeros(
            (
                imstack.shape[0],
                int(imstack.shape[1] / fac),
                int(imstack.shape[2] / fac),
                imstack.shape[3],
            ),
            "float32",
        )
        if out.shape[-1] == 3:
            # this is just an RGB image, so no need to loop over channels with PIL
            for i in range(imstack.shape[0]):
                out[i] = np.array(
                    PIL.Image.fromarray(imstack[i].astype("uint8")).resize(
                        (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS
                    )
                )
        else:
            for i in range(imstack.shape[0]):
                for j in range(imstack.shape[3]):
                    out[i, :, :, j] = np.array(
                        PIL.Image.fromarray(imstack[i, :, :, j]).resize(
                            (out.shape[2], out.shape[1]), resample=PIL.Image.LANCZOS
                        )
                    )

    elif method == "dsm":
        out = np.zeros(
            (
                imstack.shape[0],
                imstack.shape[1] // fac,
                imstack.shape[2] // fac,
                imstack.shape[3],
            ),
            "float32",
        )
        for i in range(imstack.shape[0]):
            for j in range(imstack.shape[3]):
                out[i, :, :, j] = dsm(imstack[i, :, :, j], (fac, fac))

    elif method == "nn":
        out = imstack[:, ::fac, ::fac]

    elif fac > 1:
        raise Exception("Downfac > 1. Not a valid downsampling method")

    return out


def batch_maximum(imstack):
    """Find the location of the maximum for each image in a batch."""
    maxpos = np.zeros((imstack.shape[0], 2))
    for i in range(imstack.shape[0]):
        if np.isnan(imstack[i, 0, 0]):
            maxpos[i, 0] = np.nan
            maxpos[i, 1] = np.nan
        else:
            ind = np.unravel_index(
                np.argmax(np.squeeze(imstack[i]), axis=None),
                np.squeeze(imstack[i]).shape,
            )
            maxpos[i, 0] = ind[1]
            maxpos[i, 1] = ind[0]
    return maxpos


def __initAvgMax(t, g, o, params):
    """
    Helper function for creating 3D targets
    """
    gridsize = tuple([params["nvox"]] * 3)
    g = np.reshape(g, (-1, *gridsize, 3),)

    for i in range(o.shape[0]):
        for j in range(o.shape[-1]):
            o[i, ..., j] = np.exp(
                -(
                    (g[i, ..., 1] - t[i, 1, j]) ** 2
                    + (g[i, ..., 0] - t[i, 0, j]) ** 2
                    + (g[i, ..., 2] - t[i, 2, j]) ** 2
                )
                / (2 * params["sigma"] ** 2)
            )

    return o


def batch_rgb2gray(imstack):
    """Convert to gray image-wise.

    batch dimension is first.
    """
    grayim = np.zeros((imstack.shape[0], imstack.shape[1], imstack.shape[2]), "float32")
    for i in range(grayim.shape[0]):
        grayim[i] = rgb2gray(imstack[i].astype("uint8"))
    return grayim


def return_tile(imstack, fac=2):
    """Crop a larger image into smaller tiles without any overlap."""
    height = imstack.shape[1] // fac
    width = imstack.shape[2] // fac
    out = np.zeros(
        (imstack.shape[0] * fac * fac, height, width, imstack.shape[3]), "float32"
    )
    cnt = 0
    for i in range(imstack.shape[0]):
        for j in np.arange(0, imstack.shape[1], height):
            for k in np.arange(0, imstack.shape[2], width):
                out[cnt, :, :, :] = imstack[i, j : j + height, k : k + width, :]
                cnt = cnt + 1
    return out


def tile2im(imstack, fac=2):
    """Reconstruct lagrer image from tiled data."""
    height = imstack.shape[1]
    width = imstack.shape[2]
    out = np.zeros(
        (imstack.shape[0] // (fac * fac), height * fac, width * fac, imstack.shape[3]),
        "float32",
    )
    cnt = 0
    for i in range(out.shape[0]):
        for j in np.arange(0, out.shape[1], height):
            for k in np.arange(0, out.shape[2], width):
                out[i, j : j + height, k : k + width, :] = imstack[cnt]
                cnt += 1
    return out


def cropcom(im, com, size=512):
    """Crops single input image around the coordinates com."""
    minlim_r = int(np.round(com[1])) - size // 2
    maxlim_r = int(np.round(com[1])) + size // 2
    minlim_c = int(np.round(com[0])) - size // 2
    maxlim_c = int(np.round(com[0])) + size // 2

    diff = (minlim_r, maxlim_r, minlim_c, maxlim_c)
    crop_dim = (np.max([minlim_r, 0]), maxlim_r, np.max([minlim_c, 0]), maxlim_c)

    out = im[crop_dim[0] : crop_dim[1], crop_dim[2] : crop_dim[3], :]

    dim = out.shape[2]

    # pad with zeros if region ended up outside the bounds of the original image
    if minlim_r < 0:
        out = np.concatenate(
            (np.zeros((abs(minlim_r), out.shape[1], dim)), out), axis=0
        )
    if maxlim_r > im.shape[0]:
        out = np.concatenate(
            (out, np.zeros((maxlim_r - im.shape[0], out.shape[1], dim))), axis=0
        )
    if minlim_c < 0:
        out = np.concatenate(
            (np.zeros((out.shape[0], abs(minlim_c), dim)), out), axis=1
        )
    if maxlim_c > im.shape[1]:
        out = np.concatenate(
            (out, np.zeros((out.shape[0], maxlim_c - im.shape[1], dim))), axis=1
        )

    return out, diff


def plot_markers_2d(im, markers, newfig=True):
    """Plot markers in two dimensions."""

    if newfig:
        plt.figure()
    plt.imshow((im - np.min(im)) / (np.max(im) - np.min(im)))

    for mark in range(markers.shape[-1]):
        ind = np.unravel_index(
            np.argmax(markers[:, :, mark], axis=None), markers[:, :, mark].shape
        )
        plt.plot(ind[1], ind[0], ".r")


def preprocess_3d(im_stack):
    """Easy inception-v3 style image normalization across a set of images."""
    im_stack /= 127.5
    im_stack -= 1.0
    return im_stack


def norm_im(im):
    """Normalize image."""
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def plot_markers_3d_torch(stack, nonan=True):
    """Return the 3d coordinates for each of the peaks in probability maps."""
    import torch

    n_mark = stack.shape[-1]
    index = stack.flatten(0, 2).argmax(dim=0).to(torch.int32)
    inds = unravel_index(index, stack.shape[:-1])
    if ~torch.any(torch.isnan(stack[0, 0, 0, :])) and (nonan or not nonan):
        x = inds[1]
        y = inds[0]
        z = inds[2]
    elif not nonan:
        x = inds[1]
        y = inds[0]
        z = inds[2]
        for mark in range(0, n_mark):
            if torch.isnan(stack[:, :, :, mark]):
                x[mark] = torch.nan
                y[mark] = torch.nan
                z[mark] = torch.nan
    return x, y, z


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def grid_channelwise_max(grid_):
    """Return the max value in each channel over a 3D volume.

    input--
        grid_: shape (nvox, nvox, nvox, nchannels)

    output--
        shape (nchannels,)
    """
    return np.max(np.max(np.max(grid_, axis=0), axis=0), axis=0)


def moment_3d(im, mesh, thresh=0):
    """Get the normalized spatial moments of the 3d image stack.

    inputs--
        im: 3d volume confidence map, one for each channel (marker)
            i.e. shape (nvox,nvox,nvox,nchannels)
        mesh: spatial coordinates for every position on im
        thresh: threshold applied to im before calculating moments
    """
    x = []
    y = []
    z = []
    for mark in range(im.shape[3]):
        # get normalized probabilities
        im_norm = (im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)) / np.sum(
            im[:, :, :, mark] * (im[:, :, :, mark] >= thresh)
        )
        x.append(np.sum(mesh[0] * im_norm))
        y.append(np.sum(mesh[1] * im_norm))
        z.append(np.sum(mesh[2] * im_norm))
    return x, y, z


def get_peak_inds(map_):
    """Return the indices of the peak value of an n-d map."""
    return np.unravel_index(np.argmax(map_, axis=None), map_.shape)


def get_peak_inds_multi_instance(im, n_instances, window_size=10):
    """Return top n_instances local peaks through non-max suppression."""
    bw = im == maximum_filter(im, footprint=np.ones((window_size, window_size)))
    inds = np.argwhere(bw)
    vals = im[inds[:, 0], inds[:, 1]]
    idx = np.argsort(vals)[::-1]
    return inds[idx[:n_instances], :]


def get_marker_peaks_2d(stack):
    """Return the concatenated coordinates of all peaks for each map/marker."""
    x = []
    y = []
    for i in range(stack.shape[-1]):
        inds = get_peak_inds(stack[:, :, i])
        x.append(inds[1])
        y.append(inds[0])
    return x, y


def spatial_expval(map_):
    """Calculate the spatial expected value of the input.

    Note there is probably underflow here that I am ignoring, because this
    doesn't need to be *that* accurate
    """
    map_ = map_ / np.sum(map_)
    x, y = np.meshgrid(np.arange(map_.shape[1]), np.arange(map_.shape[0]))

    return np.sum(map_ * x), np.sum(map_ * y)


def spatial_var(map_):
    """Calculate the spatial variance of the input."""
    expx, expy = spatial_expval(map_)
    map_ = map_ / np.sum(map_)
    x, y = np.meshgrid(np.arange(map_.shape[1]), np.arange(map_.shape[0]))

    return np.sum(map_ * ((x - expx) ** 2 + (y - expy) ** 2))


def spatial_entropy(map_):
    """Calculate the spatial entropy of the input."""
    map_ = map_ / np.sum(map_)
    return -1 * np.sum(map_ * np.log(map_))


def expected_value_3d(prob_map, grid_centers):
    bs, channels, h, w, d = prob_map.shape

    prob_map = prob_map.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    grid_centers = grid_centers.reshape(-1, 3)
    weighted_centers = prob_map.unsqueeze(1) * grid_centers.unsqueeze(-1)
    weighted_centers = weighted_centers.reshape(-1, h * w * d, 3, channels).sum(1)

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

    prob_map = (
        prob_map.permute(0, 2, 3, 1).reshape(bs, -1, channels).unsqueeze(2)
    )  # [bs, h*w, 1, channels]
    weighted_centers = prob_map * grid  # [bs, h*w, 2, channels]

    return weighted_centers.sum(1)  # [bs, 2, channels]


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
    grid_dist = (grid_centers.unsqueeze(-1) - markerlocs.unsqueeze(1)) ** 2
    grid_dist = grid_dist.sum(2)
    grid_dist = grid_dist.reshape(-1, channels)

    weighted_var = prob_map * grid_dist
    weighted_var = weighted_var.reshape(-1, h * w * d, channels)
    weighted_var = weighted_var.sum(1)
    return torch.mean(weighted_var, dim=-1, keepdim=True)