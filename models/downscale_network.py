import torch
import torch.nn as nn
from collections import OrderedDict

class Interpolate(nn.Module):
    def __init__(self, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=0.5, mode=self.mode, align_corners=False)
        return x

def convolveDownsample(channel_in, channel_out):
    layers = [("conv1", nn.Conv2d(channel_in, channel_out, 3, stride=1, padding=1, bias=False)),
              ("in1", nn.InstanceNorm2d(channel_out)),
              ("downsample", Interpolate(mode="bilinear"))
              ]

    return nn.Sequential(OrderedDict(layers))

class DownNet(nn.Module):
    def __init__(self, downsampling, channels=12):
        super(DownNet, self).__init__()
        list = []
        current_channels = channels
        for i in range(downsampling):
            list.append(convolveDownsample(current_channels, int(current_channels)))
            current_channels = int(current_channels)

        self.net = torch.nn.Sequential(*list)

    def forward(self, seg):
        return self.net(seg)
