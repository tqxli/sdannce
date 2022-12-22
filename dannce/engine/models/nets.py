import torch
import torch.nn as nn
from dannce.engine.models.blocks import *
from dannce.engine.data.ops import spatial_softmax, expected_value_3d


_SDANNCE_ENCDEC = [[(None, 64), (64, 128), (128, 256), (256, 512)], [(512, 256), (256, 128), (128, 64)]]
_SDANNCE_ENCDEC_COMPRESSED = [[(None, 32), (32, 64), (64, 128), (128, 256)], [(256, 128), (128, 64), (64, 32)]]


class EncDec3D(nn.Module):
    """
    ENCODER-DECODER backbone for 3D volumetric inputs
    Args:
        in_channels (int): 
        normalization (str): 
        input_shape (int)
        residual (bool)
        norm_upsampling (bool) 
        ret_enc_feat (bool)
        channel_compressed (bool)
    """
    def __init__(self, 
        in_channels, 
        normalization, 
        input_shape, 
        residual=False,
        norm_upsampling=False,
        ret_enc_feat=False,
        channel_compressed=True
    ):
        super().__init__()

        self.ret_enc_feat = ret_enc_feat

        conv_block = Res3DBlock if residual else Basic3DBlock
        deconv_block = Upsample3DBlock if norm_upsampling else BasicUpSample3DBlock
        chan_configs = _SDANNCE_ENCDEC_COMPRESSED if channel_compressed else _SDANNCE_ENCDEC

        self.encoder_res1 = conv_block(in_channels, chan_configs[0][0][1], normalization, [input_shape]*3)
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res2 = conv_block(chan_configs[0][1][0], chan_configs[0][1][1], normalization, [input_shape//2]*3)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res3 = conv_block(chan_configs[0][2][0], chan_configs[0][2][1], normalization, [input_shape//4]*3)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res4 = conv_block(chan_configs[0][3][0], chan_configs[0][3][1], normalization, [input_shape//8]*3)

        self.decoder_res3 = conv_block(chan_configs[1][0][0], chan_configs[1][0][1], normalization, [input_shape//4]*3)
        self.decoder_upsample3 = deconv_block(chan_configs[1][0][0], chan_configs[1][0][1], 2, 2, normalization, [input_shape//4]*3)
        self.decoder_res2 = conv_block(chan_configs[1][1][0], chan_configs[1][1][1], normalization, [input_shape//2]*3)
        self.decoder_upsample2 = deconv_block(chan_configs[1][1][0], chan_configs[1][1][1], 2, 2, normalization, [input_shape//2]*3)
        self.decoder_res1 = conv_block(chan_configs[1][2][0], chan_configs[1][2][1], normalization, [input_shape]*3)
        self.decoder_upsample1 = deconv_block(chan_configs[1][2][0], chan_configs[1][2][1], 2, 2, normalization, [input_shape]*3)


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

        if self.ret_enc_feat:
            return x, skips

        return x, dec_feats


class DANNCE(nn.Module):
    def __init__(
        self, 
        input_channels, 
        output_channels, 
        input_shape, 
        norm_method='layer', 
        residual=False, 
        norm_upsampling=False,
        return_inter_features=False,
        compressed=False,
        ret_enc_feat=False
    ):
        super().__init__()

        self.compressed = compressed
        self.encoder_decoder = EncDec3D(input_channels, norm_method, input_shape, residual, norm_upsampling, ret_enc_feat, channel_compressed=compressed)
        output_chan = 32 if compressed else 64
        self.output_layer = nn.Conv3d(output_chan, output_channels, kernel_size=1, stride=1, padding=0)

        self.n_joints = output_channels

        self.return_inter_features = return_inter_features
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

        if self.return_inter_features:
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
    def __init__(self, input_channels, output_channels, input_shape, n_layers=4, hidden_dims=[32, 64, 128, 256, 512], norm_method="layer"):
        super().__init__()

        assert n_layers == len(hidden_dims)-1, "Hidden dimensions do not match with the number of layers."
        conv_block = Basic2DBlock
        deconv_block = BasicUpSample2DBlock

        self.n_layers = n_layers
        self._compute_input_dims(input_shape)

        # construct layers
        self.encoder_res1 = conv_block(input_channels, 32, norm_method, self.input_dims[0])
        self.encoder_pool1 = Pool2DBlock(2)
        self.encoder_res2 = conv_block(32, 64, norm_method, self.input_dims[1])
        self.encoder_pool2 = Pool2DBlock(2)
        self.encoder_res3 = conv_block(64, 128, norm_method, self.input_dims[2])
        self.encoder_pool3 = Pool2DBlock(2)
        self.encoder_res4 = conv_block(128, 256, norm_method, self.input_dims[3])
        self.encoder_pool4 = Pool2DBlock(2)
        self.encoder_res5 = conv_block(256, 512, norm_method, self.input_dims[4])

        self.decoder_res4 = conv_block(512, 256, norm_method, self.input_dims[3])
        self.decoder_upsample4 = deconv_block(512, 256, 2, 2, norm_method, self.input_dims[3])
        self.decoder_res3 = conv_block(256, 128, norm_method, self.input_dims[2])
        self.decoder_upsample3 = deconv_block(256, 128, 2, 2, norm_method, self.input_dims[2])
        self.decoder_res2 = conv_block(128, 64, norm_method, self.input_dims[1])
        self.decoder_upsample2 = deconv_block(128, 64, 2, 2, norm_method, self.input_dims[1])
        self.decoder_res1 = conv_block(64, 32, norm_method, self.input_dims[0])
        self.decoder_upsample1 = deconv_block(64, 32, 2, 2, norm_method, self.input_dims[0])

        self.output_layer = nn.Conv2d(32, output_channels, 1, 1, 0)
     
    def _compute_input_dims(self, input_shape):
        self.input_dims = [(input_shape[0] // (2**i), input_shape[1] // (2**i)) for i in range(self.n_layers+1)]

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


def initialize_model(params, n_cams, device):
    """
    Initialize DANNCE model with params and move to GPU.
    """
    try:
        ret_enc_feat = params.get("graph_cfg", {}).get("ret_enc_feat", False)
    except:
        ret_enc_feat = False
    model_params = {
        "input_channels": (params["chan_num"] + params["depth"]) * n_cams,
        "output_channels": params["n_channels_out"],
        "norm_method": params["norm_method"],
        "input_shape": params["nvox"],
        "return_inter_features": params.get("use_features", False),
        "ret_enc_feat": ret_enc_feat 
    }

    if params["net_type"] == "dannce":
        model_params = {**model_params, "residual": False, "norm_upsampling": False}
    elif params["net_type"] == "compressed_dannce":
        model_params = {**model_params, "residual": False, "norm_upsampling": False, "compressed": True}
    elif params["net_type"] == "semi-v2v":
        model_params = {**model_params, "residual": False, "norm_upsampling": True}
    elif params["net_type"] == "v2v":
        model_params = {**model_params, "residual": True, "norm_upsampling": True}
    elif params["net_type"] == "autoencoder":
        model_params["input_channels"] = model_params["input_channels"] - 3
        model_params["output_channels"] = 3
        model_params = {**model_params, "residual": True, "norm_upsampling": True}

    model = DANNCE(**model_params)

    # model = model.to(device)
    if params["multi_gpu_train"]:
        model = nn.parallel.DataParallel(model, device_ids=params["gpu_id"])
    
    model.to(device)

    return model


def initialize_train(params, n_cams, device, logger):
    """
    Initialize model, load pretrained checkpoints if needed.
    """
    params["start_epoch"] = 1

    if params["train_mode"] == "new":
        logger.info("*** Traininig from scratch. ***")
        model = initialize_model(params, n_cams, device)
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)

    elif params["train_mode"] == "finetune":
        logger.info("*** Finetuning from {}. ***".format(params["dannce_finetune_weights"]))
        checkpoints = torch.load(params["dannce_finetune_weights"])
        model = initialize_model(params, n_cams, device)

        state_dict = checkpoints["state_dict"]


        # ckpt_channel_num = state_dict["encoder_decoder.encoder_res1.block.0.weight"].shape[0]
        # if ckpt_channel_num != params["n_views"]*params["chan_num"]:
        #     state_dict.pop("encoder_decoder.encoder_res1.block.0.weight", None)
        #     state_dict.pop("encoder_decoder.encoder_res1.block.0.bias", None)

        # replace final output layer if do not match with the checkpoint
        # try:
        ckpt_channel_num = state_dict["output_layer.weight"].shape[0]
        if ckpt_channel_num != params["n_channels_out"]:
            state_dict.pop("output_layer.weight", None)
            state_dict.pop("output_layer.bias", None)
        model.load_state_dict(state_dict, strict=False)
        # except:
        #     model.load_state_dict(state_dict, strict=False)

        # for name, param in model.named_parameters():
        #     if 'encoder_decoder.encoder' in name:
        #         param.requires_grad = False
 
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)
    
    elif params["train_mode"] == "continued":
        logger.info("*** Resume training from {}. ***".format(params["dannce_finetune_weights"]))
        checkpoints = torch.load(params["dannce_finetune_weights"])
        
        # ensure the same architecture
        model = initialize_model(checkpoints["params"], n_cams, device)
        model.load_state_dict(checkpoints["state_dict"], strict=True)

        model_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(model_params)
        optimizer.load_state_dict(checkpoints["optimizer"])

        # specify the start epoch
        params["start_epoch"] = checkpoints["epoch"]
    
    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, params["lr_scheduler"]["type"])
        lr_scheduler = lr_scheduler_class(optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True)
        logger.info("Using learning rate scheduler: {}".format(params["lr_scheduler"]["type"]))
    
    return model, optimizer, lr_scheduler


def initialize_com_model(params, device):
    model = COMNet(params["n_channels_in"], params["n_channels_out"], params["input_shape"]).to(device)
    return model


def initialize_com_train(params, device, logger):
    if params["train_mode"] == "new":
        logger.info("*** Traininig from scratch. ***")
        model = initialize_com_model(params, device)
        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)

    elif params["train_mode"] == "finetune":
        logger.info("*** Finetuning from {}. ***".format(params["com_finetune_weights"]))
        checkpoints = torch.load(params["com_finetune_weights"])
        model = initialize_com_model(params, device)

        state_dict = checkpoints["state_dict"]
        # replace final output layer if do not match with the checkpoint
        ckpt_channel_num = state_dict["output_layer.weight"].shape[0]
        if ckpt_channel_num != params["n_channels_out"]:
            state_dict.pop("output_layer.weight", None)
            state_dict.pop("output_layer.bias", None)

        model.load_state_dict(state_dict, strict=False)

        model_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(model_params, lr=params["lr"], eps=1e-7)
    
    elif params["train_mode"] == "continued":
        logger.info("*** Resume training from {}. ***".format(params["dannce_finetune_weights"]))
        checkpoints = torch.load(params["dannce_finetune_weights"])
        
        # ensure the same architecture
        model = initialize_com_model(checkpoints["params"], device)
        model.load_state_dict(checkpoints["state_dict"], strict=True)

        model_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.Adam(model_params)
        optimizer.load_state_dict(checkpoints["optimizer"])

        # specify the start epoch
        params["start_epoch"] = checkpoints["epoch"]
    
    lr_scheduler = None
    if params["lr_scheduler"] is not None:
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, params["lr_scheduler"]["type"])
        lr_scheduler = lr_scheduler_class(optimizer=optimizer, **params["lr_scheduler"]["args"], verbose=True)
        logger.info("Using learning rate scheduler.")
    
    return model, optimizer, lr_scheduler 

if __name__ == "__main__":
    model_params = {
        "input_channels": 18,
        "output_channels": 23,
        "norm_method": 'batch',
        "input_shape": 80
    }
    model_params = {**model_params, "residual": False, "norm_upsampling": False}
    model = DANNCE(**model_params)

    input_shape = [128, 80, 8] # encoder-decoder downsamples for 3 times which force input dimension to be divisble by 2**3 = 8
    inputs = torch.randn(1, 18, *input_shape)
    (x_coord, y_coord, z_coord) = torch.meshgrid(torch.arange(input_shape[0]), torch.arange(input_shape[1]), torch.arange(input_shape[2]))
    grid_centers = torch.stack((x_coord, y_coord, z_coord), axis=0).unsqueeze(0)
    grid_centers = grid_centers.reshape(*grid_centers.shape[:2], -1)

    _, heatmaps = model(inputs, grid_centers)
    print(heatmaps.shape)