import torch
import torch.nn as nn
from models.igam_module import IGAM

class MBGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MBGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.igam = IGAM(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.igam(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out += residual
        return torch.relu(out)
