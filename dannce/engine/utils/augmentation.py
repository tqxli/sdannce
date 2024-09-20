import numpy as np
import torch


def random_rotate(X, y_3d, aux=None):
    def rot90(X):
        X = X.permute(1, 0, 2, 3)
        return X.flip(1)

    def rot180(X):
        return X.flip(0).flip(1)

    rot = np.random.choice(np.arange(4), 1)
    for j in range(X.shape[0]):
        if rot == 0:
            pass
        elif rot == 1:
            # Rotate180
            X[j], y_3d[j] = rot180(X[j]), rot180(y_3d[j])
            if aux is not None:
                aux[j] = rot180(aux[j])
        elif rot == 2:
            # Rotate90
            X[j], y_3d[j] = rot90(X[j]), rot90(y_3d[j])
            if aux is not None:
                aux[j] = rot90(aux[j])
        elif rot == 3:
            # Rotate -90/270
            X[j], y_3d[j] = rot180(rot90(X[j])), rot180(rot90(y_3d[j]))
            if aux is not None:
                aux[j] = rot180(rot90(aux[j]))
    return X, y_3d, aux


def apply_random_transforms(volumes, grids, aux=None):
    grids = grids.reshape(grids.shape[0], 80, 80, 80, 3)

    volumes, grids, aux = random_rotate(volumes, grids, aux)
    grids = grids.reshape(grids.shape[0], -1, 3)
    return volumes, grids, aux


def construct_augmented_batch(volumes, grids, aux=None, copies_per_sample=1):
    copies = []
    n_samples = volumes.shape[0]
    for i in range(n_samples):
        copies += [
            apply_random_transforms(
                volumes[i].clone().unsqueeze(0),
                grids[i].clone().unsqueeze(0),
                aux[i].clone().unsqueeze(0) if aux is not None else aux,
            )
            for _ in range(copies_per_sample)
        ]

    volumes = torch.cat([copy[0] for copy in copies], dim=0)
    grids = torch.cat([copy[1] for copy in copies], dim=0)
    aux = torch.cat([copy[2] for copy in copies], dim=0) if aux is not None else aux

    return volumes, grids, aux