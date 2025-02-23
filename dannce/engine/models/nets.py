import os
from typing import Dict, List, Literal

import torch
import torch.nn as nn
from loguru import logger

from dannce.engine.models.blocks import *
from dannce.engine.models.posegcn.nets import PoseGCN
from dannce.engine.utils.image import expected_value_3d, spatial_softmax

_SDANNCE_ENCDEC = [
    [(None, 64), (64, 128), (128, 256), (256, 512)],
    [(512, 256), (256, 128), (128, 64)],
]
_SDANNCE_ENCDEC_COMPRESSED = [
    [(None, 32), (32, 64), (64, 128), (128, 256)],
    [(256, 128), (128, 64), (64, 32)],
]


class EncDec3D(nn.Module):
    """
    ENCODER-DECODER backbone for 3D volumetric inputs
    Args:
        in_channels (int): number of input channels (keypoints) 
        normalization (str): normalization method
        input_shape (int): (H, W, D) shape of input volume
        residual (bool)
        norm_upsampling (bool) 
        return_enc_feats (bool)
        channel_compressed (bool)
    """

    def __init__(
        self,
        in_channels: int,
        normalization: Literal["batch", "instance", "layer"],
        input_shape: int,
        residual: bool = False,
        norm_upsampling: bool = False,
        return_enc_feats: bool = False,
        channel_compressed: bool = True,
    ):
        super().__init__()

        self.return_enc_feats = return_enc_feats

        conv_block = Res3DBlock if residual else Basic3DBlock
        deconv_block = Upsample3DBlock if norm_upsampling else BasicUpSample3DBlock
        chan_configs = (
            _SDANNCE_ENCDEC_COMPRESSED if channel_compressed else _SDANNCE_ENCDEC
        )

        for idx, (chans_in, chans_out) in enumerate(chan_configs[0]):
            if idx == 0 and chans_in is None:
                chans_in = in_channels
            setattr(
                self,
                f"encoder_res{idx+1}",
                conv_block(
                    chans_in, chans_out, normalization, [input_shape // 2 ** idx] * 3
                ),
            )
            if idx != len(chan_configs[0]) - 1:
                setattr(self, f"encoder_pool{idx+1}", Pool3DBlock(2))

        for idx, (chans_in, chans_out) in enumerate(chan_configs[1]):
            name_idx = len(chan_configs[1]) - idx
            setattr(
                self,
                f"decoder_res{name_idx}",
                conv_block(
                    chans_in,
                    chans_out,
                    normalization,
                    [input_shape // 2 ** (name_idx - 1)] * 3,
                ),
            )
            setattr(
                self,
                f"decoder_upsample{name_idx}",
                deconv_block(
                    chans_in,
                    chans_out,
                    2,
                    2,
                    normalization,
                    [input_shape // 2 ** (name_idx - 1)] * 3,
                ),
            )

    def forward(self, x):
        skips, dec_feats = [], []
        # encoder
        x = self.encoder_res1(x)
        skip_x1 = x
        skips.append(skip_x1)
        x = self.encoder_pool1(x)

        x = self.encoder_res2(x)
        skip_x2 = x
        skips.append(skip_x2)
        x = self.encoder_pool2(x)

        x = self.encoder_res3(x)
        skip_x3 = x
        skips.append(skip_x3)
        x = self.encoder_pool3(x)

        x = self.encoder_res4(x)

        # decoder with skip connections
        x = self.decoder_upsample3(x)
        x = self.decoder_res3(torch.cat([x, skip_x3], dim=1))
        dec_feats.append(x)
        x = self.decoder_upsample2(x)
        x = self.decoder_res2(torch.cat([x, skip_x2], dim=1))
        dec_feats.append(x)
        x = self.decoder_upsample1(x)
        x = self.decoder_res1(torch.cat([x, skip_x1], dim=1))
        dec_feats.append(x)

        if self.return_enc_feats:
            return x, skips

        return x, dec_feats


class DANNCE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_shape: int,
        norm_method: Literal["batch", "instance", "layer"] = "layer",
        residual: bool = False,
        norm_upsampling: bool = False,
        return_features: bool = False,
        compressed: bool = False,
        return_enc_feats: bool = False,
    ):
        super().__init__()

        self.compressed = compressed
        self.encoder_decoder = EncDec3D(
            input_channels,
            norm_method,
            input_shape,
            residual,
            norm_upsampling,
            return_enc_feats,
            channel_compressed=compressed,
        )
        output_chan = 32 if compressed else 64
        self.output_layer = nn.Conv3d(
            output_chan, output_channels, kernel_size=1, stride=1, padding=0
        )

        self.n_joints = output_channels

        self.return_features = return_features
        self._initialize_weights()

    def forward(self, volumes, grid_centers):
        """
        volumes: Tensor [batch_size, C, H, W, D]
        grid_centers: [batch_size, nvox**3, 3]
        """
        volumes, inter_features = self.encoder_decoder(volumes)
        heatmaps = self.output_layer(volumes)

        if grid_centers is not None:
            softmax_heatmaps = spatial_softmax(heatmaps)
            coords = expected_value_3d(softmax_heatmaps, grid_centers)
        else:
            coords = None

        if self.return_features:
            return coords, heatmaps, inter_features
        for f in inter_features:
            del f
        return coords, heatmaps, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


class COMNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_shape: int,
        n_layers: int = 4,
        hidden_dims: List[int] = [32, 64, 128, 256, 512],
        norm_method: Literal["batch", "instance", "layer"] = "layer",
    ):
        super().__init__()

        assert (
            n_layers == len(hidden_dims) - 1
        ), "Hidden dimensions do not match with the number of layers."
        conv_block = Basic2DBlock
        deconv_block = BasicUpSample2DBlock

        self.n_layers = n_layers
        self._compute_dims(input_shape)

        for idx, (chans_in, chans_out) in enumerate(self.chan_config[0]):
            if idx == 0 and chans_in is None:
                chans_in = input_channels
            setattr(
                self,
                f"encoder_res{idx+1}",
                conv_block(chans_in, chans_out, norm_method, self.input_dims[idx]),
            )
            if idx != len(self.chan_config[0]) - 1:
                setattr(self, f"encoder_pool{idx+1}", Pool2DBlock(2))

        for idx, (chans_in, chans_out) in enumerate(self.chan_config[1]):
            name_idx = len(self.chan_config[1]) - idx
            setattr(
                self,
                f"decoder_res{name_idx}",
                conv_block(
                    chans_in, chans_out, norm_method, self.input_dims[name_idx - 1]
                ),
            )
            setattr(
                self,
                f"decoder_upsample{name_idx}",
                deconv_block(
                    chans_in,
                    chans_out,
                    2,
                    2,
                    norm_method,
                    self.input_dims[name_idx - 1],
                ),
            )

        self.output_layer = nn.Conv2d(32, output_channels, 1, 1, 0)

    def _compute_dims(self, input_shape):
        self.input_dims = [
            (input_shape[0] // (2 ** i), input_shape[1] // (2 ** i))
            for i in range(self.n_layers + 1)
        ]
        self.chan_config = [
            [(None, 32), (32, 64), (64, 128), (128, 256), (256, 512)],
            [(512, 256), (256, 128), (128, 64), (64, 32)],
        ]

    def forward(self, x):
        # encoder
        x = self.encoder_res1(x)
        skip_x1 = x
        x = self.encoder_pool1(x)

        x = self.encoder_res2(x)
        skip_x2 = x
        x = self.encoder_pool2(x)

        x = self.encoder_res3(x)
        skip_x3 = x
        x = self.encoder_pool3(x)

        x = self.encoder_res4(x)
        skip_x4 = x
        x = self.encoder_pool4(x)

        x = self.encoder_res5(x)

        # decoder with skip connections
        x = self.decoder_upsample4(x)
        x = self.decoder_res4(torch.cat([x, skip_x4], dim=1))

        x = self.decoder_upsample3(x)
        x = self.decoder_res3(torch.cat([x, skip_x3], dim=1))

        x = self.decoder_upsample2(x)
        x = self.decoder_res2(torch.cat([x, skip_x2], dim=1))

        x = self.decoder_upsample1(x)
        x = self.decoder_res1(torch.cat([x, skip_x1], dim=1))

        x = self.output_layer(x)

        return x


def _initialize_dannce_backbone(
    params: Dict, n_cams: int,
):
    # retrieve parameters needed for initializing the 3D backbone
    model_params = {
        # architecture
        "compressed": params["net_type"] == "compressed_dannce",
        "residual": params.get("residual", False),
        "norm_upsampling": params.get("norm_upsampling", False),
        # I/O shape
        "input_channels": (params["chan_num"] + params["depth"]) * n_cams,
        "output_channels": params["n_channels_out"],
        "norm_method": params["norm_method"],
        "input_shape": params["nvox"],
        # returns
        "return_features": params.get("use_features", False),
        "return_enc_feats": params["graph_cfg"].get("return_enc_feats", False)
        if "graph_cfg" in params and params["graph_cfg"] is not None
        else False,
    }

    # initialize the backbone
    model = DANNCE(**model_params)
    return model


def _initialize_dannce(
    params: Dict, n_cams: int,
):
    return _initialize_dannce_backbone(params, n_cams)


def _initialize_sdannce(
    params: Dict, n_cams: int,
):
    # specific parameters for sdannce
    sdannce_model_params = params["graph_cfg"]

    # initialize the 3D backbone
    pose_generator = _initialize_dannce_backbone(params, n_cams)

    # add additional layers for pose refinement
    model = PoseGCN(params, sdannce_model_params, pose_generator)
    return model


def _initialize_com(params: Dict, n_cams: int):
    model = COMNet(params["chan_num"], params["n_channels_out"], params["input_shape"])
    return model


def initialize_model(
    params: Dict, n_cams: int, device: torch.device, model_type="dannce",
):
    initialize_fcn = globals()[f"_initialize_{model_type}"]
    model = initialize_fcn(params, n_cams)

    if params.get("multi_gpu_train", False):
        model = nn.DataParallel(model, device_ids=params["gpu_id"])
    return model.to(device)


def checkpoint_weights_type(state_dict: Dict):
    input_weight_name = "encoder_decoder.encoder_res1.block.0.weight"
    output_weight_name = "output_layer.weight"

    if input_weight_name in state_dict:
        is_sdannce_weights = False
        # weight shape should be [32, 3*n_cams, 3, 3, 3]
        checkpoint_input_size = state_dict[input_weight_name].shape[1]
        checkpoint_output_size = state_dict[output_weight_name].shape[0]
    elif f"pose_generator.{input_weight_name}" in state_dict:
        is_sdannce_weights = True
        input_weight_name = f"pose_generator.{input_weight_name}"
        output_weight_name = f"pose_generator.{output_weight_name}"
        checkpoint_input_size = state_dict[input_weight_name].shape[1]
        checkpoint_output_size = state_dict[output_weight_name].shape[0]
    else:
        raise ValueError("Invalid checkpoint format.")

    return (
        is_sdannce_weights,
        checkpoint_input_size,
        checkpoint_output_size,
        input_weight_name,
        output_weight_name,
    )


def load_pretrained_weights(
    model: nn.Module, checkpoint_path: str, skip_io_check: bool = False,
):
    """
    Load pretrained weights into (s)DANNCE model.
    """
    assert checkpoint_path is not None and os.path.exists(
        checkpoint_path
    ), f"Checkpoint not found: {checkpoint_path}"
    logger.info(f"Loading pretrained weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)["state_dict"]

    if skip_io_check:
        model.load_state_dict(state_dict, strict=False)
        return model

    # check whether input & output dimensions mismatch
    (
        is_sdannce_weights,
        checkpoint_input_size,
        checkpoint_output_size,
        input_weight_name,
        output_weight_name,
    ) = checkpoint_weights_type(state_dict)
    (
        is_sdannce_model,
        model_input_size,
        model_output_size,
        _,
        _,
    ) = checkpoint_weights_type(model.state_dict())

    # pop mismatch weights from checkpoint
    if checkpoint_input_size != model_input_size:
        logger.warning(
            f"Input dimension mismatch: checkpoint ({checkpoint_input_size}) vs model ({model_input_size}). Re-initializing weights."
        )
        state_dict.pop(input_weight_name, None)
        state_dict.pop(input_weight_name.replace("weight", "bias"), None)

    if checkpoint_output_size != model_output_size:
        logger.warning(
            f"Output dimension mismatch: checkpoint ({checkpoint_output_size}) vs model ({model_output_size}). Re-initializing weights."
        )
        state_dict.pop(output_weight_name, None)
        state_dict.pop(output_weight_name.replace("weight", "bias"), None)

    # load weights
    if is_sdannce_weights == is_sdannce_model:
        model.load_state_dict(state_dict, strict=False)
    elif is_sdannce_model and not is_sdannce_weights:
        model.pose_generator.load_state_dict(state_dict, strict=False)
    elif not is_sdannce_model and is_sdannce_weights:
        logger.warning(
            "Loading weights from SDANNCE to DANNCE! Check if this is intended."
        )
        state_dict = {
            k.replace("pose_generator.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)

    return model


def initialize_train(
    params: Dict,
    n_cams: int,
    device: torch.device,
    model_type=Literal["dannce", "sdannce"],
):
    """
    Initialize model and load pretrained weights if 'dannce_finetune_weights' is specified in config.
    """
    params["start_epoch"] = 1

    train_mode = params["train_mode"]
    assert train_mode in [
        "new",
        "finetune",
        "continued",
    ], f"Invalid training mode: {train_mode}"

    # initialize model

    model = initialize_model(params, n_cams, device, model_type)

    # load pretrained weights
    if train_mode == "finetune" or train_mode == "continued":
        checkpoint_path = params.get("dannce_finetune_weights", None)
        model = load_pretrained_weights(model, checkpoint_path)

    model_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model_params) / 1e6:.2f}M"
    )

    optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)

    if params["train_mode"] == "continued":
        logger.info(
            "*** Resume training from {} ***".format(params["dannce_finetune_weights"])
        )
        checkpoints = torch.load(params["dannce_finetune_weights"])
        optimizer = torch.optim.Adam(model_params)
        optimizer.load_state_dict(checkpoints["optimizer"])
        params["start_epoch"] = checkpoints["epoch"]

    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(
            torch.optim.lr_scheduler, params["lr_scheduler"]["type"]
        )
        lr_scheduler = lr_scheduler_class(
            optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True
        )
        logger.info(
            "Using learning rate scheduler: {}".format(params["lr_scheduler"]["type"])
        )

    return model, optimizer, lr_scheduler


def initialize_prediction(
    params: Dict, n_cams: int, device: torch.device, model_type="dannce",
):
    """
    Initialize model for prediction and load pretrained weights.
    """
    model = initialize_model(params, n_cams, device, model_type)

    # one should expect no mismatch in input/output dimensions
    checkpoint_path = params.get("dannce_predict_model", None)
    model = load_pretrained_weights(model, checkpoint_path, skip_io_check=True)
    model.eval()
    return model


def load_pretrained_com_weights(
    params: Dict, model: nn.Module, checkpoint_path: str, skip_io_check: bool = False,
):
    """
    Load pretrained weights for COM model.
    """
    assert checkpoint_path is not None and os.path.exists(
        checkpoint_path
    ), f"Checkpoint not found: {checkpoint_path}"
    logger.info(f"Loading pretrained weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path)["state_dict"]

    if skip_io_check:
        model.load_state_dict(state_dict, strict=False)
        return model

    ckpt_channel_num = state_dict["output_layer.weight"].shape[0]
    if ckpt_channel_num != params["n_channels_out"]:
        state_dict.pop("output_layer.weight", None)
        state_dict.pop("output_layer.bias", None)

    model.load_state_dict(state_dict, strict=True)
    return model


def initialize_com_train(
    params: Dict, device: torch.device,
):
    """
    Initialize COM model and load pretrained weights if 'com_finetune_weights' is specified in config.
    """
    train_mode = params["train_mode"]
    assert train_mode in ["new", "finetune",], f"Invalid training mode: {train_mode}"

    # initialize model
    model = initialize_model(params, -1, device, "com")

    # load pretrained weights
    if train_mode == "finetune":
        checkpoint_path = params.get("com_finetune_weights", None)
        if checkpoint_path is None:
            logger.warning("No pretrained weights specified for finetuning mode.")
        model = load_pretrained_com_weights(params, model, checkpoint_path)

    model_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model_params) / 1e6:.2f}M"
    )
    optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)

    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(
            torch.optim.lr_scheduler, params["lr_scheduler"]["type"]
        )
        lr_scheduler = lr_scheduler_class(
            optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True
        )
        logger.info(
            "Using learning rate scheduler: {}".format(params["lr_scheduler"]["type"])
        )
    return model, optimizer, lr_scheduler
