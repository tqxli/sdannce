import torch
import torch.nn as nn

from dannce.engine.models.posegcn.gcn_blocks import (SemGraphConv, _GraphConv,
                                                     _ResGraphConv)
from dannce.engine.models.posegcn.utils import *
from dannce.engine.skeletons.utils import load_body_profile
from dannce.engine.utils.image import max_coord_3d

NODES_GROUP = [[i] for i in range(23)]
TEMPORAL_FLOW = np.array(
    [0, 4, 9, 13, 17, 21]
)  # restrict the flows along temporal dimension


class PoseGCN(nn.Module):
    def __init__(
        self, params, model_params, pose_generator,
    ):
        super(PoseGCN, self).__init__()

        # pose estimation backbone
        self.pose_generator = pose_generator

        # use relative voxel coordinates instead of absolute 3D world coordinates
        self.use_relpose = model_params.get("use_relpose", True)

        # GCN architecture
        self.input_dim = input_dim = model_params.get("input_dim", 3)
        self.hid_dim = hid_dim = model_params.get("hidden_dim", 128)
        self.n_layers = n_layers = model_params.get("n_layers", 3)
        self.base_block = base_block = model_params.get("base_block", "sem")
        gconv_block = SemGraphConv
        self.norm_type = norm_type = model_params.get("norm_type", "batch")
        self.dropout = dropout = model_params.get("dropout", None)

        self.fuse_dim = fuse_dim = model_params.get("fuse_dim", 256)

        # skeletal graph construction
        self.n_instances = n_instances = model_params.get("n_instances", 1)
        self.n_joints = n_joints = params[
            "n_channels_out"
        ]  # num of nodes = n_instance * n_joints
        self.t_dim = t_dim = params.get("temporal_chunk_size", 1)
        self.t_flow = t_flow = model_params.get("t_flow", TEMPORAL_FLOW)
        inter_social = model_params.get("inter_social", False)
        self.social = (n_instances > 1) and inter_social

        # adjacency matrix
        body_profile = load_body_profile(params.get("skeleton", "rat23"))
        self.adj = adj = build_adj_mx_from_edges(
            social=self.social,
            t_dim=t_dim,
            t_flow=t_flow,
            num_joints=n_joints,
            edge=body_profile["limbs"],
        )

        # construct GCN layers
        self.use_features = use_features = model_params.get("use_features", False)

        # self.gconv_input = _GraphConv(adj, input_dim, hid_dim, dropout, base_block=base_block, norm_type=norm_type)
        self.gconv_input = []
        try:
            self.compressed = self.pose_generator.compressed
        except:
            self.compressed = self.pose_generator.posenet.compressed

        # in multi-scale features extracted from decoder layers
        if use_features:
            self.multi_scale_fdim = 128 + 64 + 32 if self.compressed else 256 + 128 + 64
            gconv_inputdim = input_dim + fuse_dim
            self.fusion_layer = nn.Conv1d(
                self.multi_scale_fdim, fuse_dim, kernel_size=1
            )

            self.gconv_input.append(
                _GraphConv(adj, gconv_inputdim, hid_dim, dropout, base_block, norm_type)
            )
        else:
            self.gconv_input.append(
                _GraphConv(adj, input_dim, hid_dim, dropout, base_block, norm_type)
            )

        gconv_layers = []
        for _ in range(n_layers):
            gconv_layers.append(
                _ResGraphConv(
                    adj,
                    hid_dim,
                    hid_dim,
                    hid_dim,
                    dropout,
                    base_block=base_block,
                    norm_type=norm_type,
                )
            )

        self.gconv_input = nn.Sequential(*self.gconv_input)
        self.gconv_layers = nn.Sequential(*gconv_layers)
        self.gconv_output = gconv_block(hid_dim, input_dim, adj)

        self.use_residual = model_params.get("use_residual", True)

    def forward(self, volumes, grid_centers):
        # initial pose generation from encoder-decoder
        init_poses, heatmaps, inter_features = self.pose_generator(
            volumes, grid_centers
        )

        # refine pose estimations using GCN
        final_poses = self.inference(init_poses, grid_centers, heatmaps, inter_features)

        if self.use_residual:
            final_poses += init_poses

        return init_poses, final_poses, heatmaps

    def inference(self, init_poses, grid_centers, heatmaps=None, inter_features=None):
        coord_grids = (
            max_coord_3d(heatmaps).unsqueeze(2).unsqueeze(2)
        )  # [B, 23, 1, 1, 3]

        # normalize the absolute 3D coordinates to relative voxel coordinates
        if self.use_relpose:
            com3d = torch.mean(grid_centers, dim=1).unsqueeze(-1)  # [N, 3, 1]
            nvox = round(grid_centers.shape[1] ** (1 / 3))
            vsize = (grid_centers[0, :, 0].max() - grid_centers[0, :, 0].min()) / nvox
            init_poses = (init_poses - com3d) / vsize

        x = init_poses.transpose(2, 1).contiguous()  # [B, 23, 3]

        if self.social:
            # whether jointly optimize both sets of keypoints
            x = x.reshape(
                init_poses.shape[0] // self.n_instances, -1, 3
            ).contiguous()  # [n, 46, 3] or [n, 23, 3]
        else:
            # treat separately
            x = x.reshape(
                init_poses.shape[0] * self.n_instances, -1, 3
            ).contiguous()  # [x*2, 23, 3]

        # if inputs are across time
        x = x.reshape(
            -1, self.t_dim * x.shape[1], x.shape[2]
        ).contiguous()  # [n, t_dim*23, 3]

        # use multi-scale features
        if self.use_features:
            f3 = (
                F.grid_sample(inter_features[0], coord_grids, align_corners=True)
                .squeeze(-1)
                .squeeze(-1)
            )
            f2 = (
                F.grid_sample(inter_features[1], coord_grids, align_corners=True)
                .squeeze(-1)
                .squeeze(-1)
            )
            f1 = (
                F.grid_sample(inter_features[2], coord_grids, align_corners=True)
                .squeeze(-1)
                .squeeze(-1)
            )

            f = self.fusion_layer(torch.cat((f3, f2, f1), dim=1))
            x = torch.cat((f.permute(0, 2, 1), x), dim=-1)

        x = self.gconv_input(x)
        x = self.gconv_layers(x)
        x = self.gconv_output(x)

        x = (
            x.reshape(init_poses.shape[0], -1, 3).transpose(2, 1).contiguous()
        )  # [n, 3, 23]

        final_poses = x
        return final_poses


if __name__ == "__main__":
    model = PoseGCN()

    input = torch.randn(5, 23, 3)
    print("Input: ", input.shape)
    output = model(input)
    print("Output: ", output.shape)
