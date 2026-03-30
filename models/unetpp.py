import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(UNetPlusPlus, self).__init__()
        filters = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])
        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def _upsample_to(self, tensor, reference):
        return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample_to(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample_to(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample_to(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample_to(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample_to(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample_to(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample_to(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample_to(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample_to(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_to(x1_3, x0_0)], dim=1))

        return self.final(x0_4)
