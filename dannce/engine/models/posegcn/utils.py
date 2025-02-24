import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from dannce.engine.skeletons.utils import load_body_profile

NUM_JOINTS = 23
EDGE = load_body_profile("rat23")["limbs"]
TEMPORAL_FLOW = np.arange(
    NUM_JOINTS
)  # np.array([0, 4, 9, 13, 17, 21]) # restrict the flows along temporal dimension

# using to build edge using GCN
def build_adj_mx_from_edges(
    num_joints=NUM_JOINTS, edge=EDGE, social=False, t_dim=1, t_flow=None,
):
    if t_flow is None:
        t_flow = np.arange(num_joints)

    if social:
        inter = np.stack(
            (np.arange(num_joints), np.arange(num_joints) + num_joints), axis=-1
        )
        edge = np.concatenate((edge, edge + num_joints, inter), axis=0)
        num_joints *= 2

    if t_dim > 1:
        inter, intra = [], []
        for i in range(t_dim - 1):
            inter.append(
                np.stack(t_flow + i * num_joints, t_flow + (i + 1) * num_joints),
                axis=-1,
            )
            # inter.append(np.stack((np.arange(num_joints)+i*num_joints, np.arange(num_joints)+(i+1)*num_joints), axis=-1))
        for i in range(t_dim):
            intra.append(edge + num_joints * i)
        inter = np.concatenate(inter, axis=0)
        intra = np.concatenate(intra, axis=0)
        edge = np.concatenate((inter, intra), axis=0)

        num_joints *= t_dim

    return adj_mx_from_edges(num_joints, edge, False)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    # breakpoint()
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = (
        adj_mx
        + adj_mx.T.multiply(adj_mx.T > adj_mx)
        - adj_mx.multiply(adj_mx.T > adj_mx)
    )
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def adj_mx_from_edges_social(num_pts, edges, sparse=True):
    mx1 = adj_mx_from_edges(num_pts, edges, sparse)
    mx2 = adj_mx_from_edges(num_pts, edges, sparse)
    bl = np.zeros_like(mx1)
    tr = np.eye(num_pts)

    mx = np.concatenate((np.concatenate((mx1, bl), 0), np.concatenate((tr, mx2))), 1)
    return mx
