import torch
import torch.nn as nn
from models.mbg_conv import MBGBlock
from models.igam_module import IGAM

class EGANet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(EGANet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.main_branch = nn.Sequential(
            MBGBlock(32, 64),
            MBGBlock(64, 128),
            MBGBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.mask_branch = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.initial(x)
        mask = self.mask_branch(x)            # shape: [B, 1, H, W]
        feat = self.main_branch(x)            # shape: [B, 256, 1, 1]
        feat = feat * mask.mean(dim=[2, 3], keepdim=True)
        feat = feat.view(feat.size(0), -1)    # flatten
        out = self.classifier(feat)
        return out
