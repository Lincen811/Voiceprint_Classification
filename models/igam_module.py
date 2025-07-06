import torch
import torch.nn as nn
import torch.nn.functional as F

class IGAM(nn.Module):
    def __init__(self, channels):
        super(IGAM, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        proj_query = self.query_conv(x).view(batch, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch, -1, h * w)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)
        out = self.gamma * out + x

        spatial_mask = self.spatial(x)
        return out * spatial_mask
