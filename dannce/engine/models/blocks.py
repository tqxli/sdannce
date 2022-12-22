import torch.nn as nn
import torch.nn.functional as F
from .normalization import LayerNormalization

NORMALIZATION_MODES = {
    'batch': nn.BatchNorm3d,
    'instance': nn.InstanceNorm3d,
    'layer': LayerNormalization,# nn.LayerNorm
}

class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_method, input_shape, kernel_size=3):
        super(Basic3DBlock, self).__init__()

        self.normalization = NORMALIZATION_MODES[norm_method]
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_method, input_shape):
        super(Res3DBlock, self).__init__()
        self.normalization = NORMALIZATION_MODES[norm_method]
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class BasicUpSample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, norm_method, input_shape):
        super(BasicUpSample3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0)
        )
    
    def forward(self, x):
        return self.block(x)

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, norm_method, input_shape):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.normalization = NORMALIZATION_MODES[norm_method]
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_method, input_shape, kernel_size=3):
        super(Basic2DBlock, self).__init__()

        self.normalization = NORMALIZATION_MODES[norm_method]
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            self.normalization([out_planes, *input_shape]) if norm_method == 'layer' else self.normalization(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

class Pool2DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

class BasicUpSample2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, norm_method, input_shape):
        super(BasicUpSample2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0)
        )
    
    def forward(self, x):
        return self.block(x)