import math
from torch import nn
import torch


class edsr(nn.Module):
    """
        The network architecture is same as EDSR (official implementation: https://github.com/thstkdgus35/EDSR-PyTorch/blob/master/src/model/edsr.py)
        but we DO NOT use "MeanShift" (i.e. sub_mean and add_mean),
        and we DO use "tanh" to ensure the output pixels are [-1, 1].
    """

    def __init__(self, scale_factor=4, num_resblocks=32, num_filters=256):
        """
            Refer to EDSR paper's Figure 3.
        """
        super(edsr, self).__init__()
        self.isBN = False
        self.isAct = False
        self.residual_scale = 0.1
        self.scale_factor = scale_factor
        self.num_filters = num_filters
        # Conv1
        self.conv1 = nn.Conv2d(3, self.num_filters, kernel_size=3, padding=1)
        # ResBlock
        self.res_blocks = self.Make_ResBlocks(num_resblocks)
        # Conv2
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1)
        # Upsample
        self.upsampler = self.Make_Upsampler()
        # Conv3
        self.conv3 = nn.Conv2d(self.num_filters, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(self.res_blocks(x)) + x
        out = self.conv3(self.upsampler(res))
        return torch.tanh(out)  # ensure -1<=out<=1

    def Make_ResBlocks(self, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers += [ResBlock(self.num_filters, self.residual_scale, self.isBN)]
        return nn.Sequential(*layers)

    def Make_Upsampler(self):
        num_upsample_blocks = int(math.log(self.scale_factor, 2))
        layers = []
        for _ in range(num_upsample_blocks):
            layers += [UpsampleBlock(self.num_filters, 2, self.isAct)]
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, channels, res_scale, isBN=False):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        if isBN:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        return x + self.res_scale * self.residual(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale, isAct=False):
        super(UpsampleBlock, self).__init__()
        self.isAct = isAct
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        if self.isAct:
            x = nn.ReLU(inplace=True)(x)
        return x
