import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class input_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(input_block, self).__init__()
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_block, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_block(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_block, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        :param x1: Current layer
        :param x2: Copy from previous layer
        :return: Concatenated layer
        """
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class output_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(output_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.input = input_block(n_channels, 64)

        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 512)

        self.up1 = up_block(1024, 256)
        self.up2 = up_block(512, 128)
        self.up3 = up_block(256, 64)
        self.up4 = up_block(128, 64)

        self.output = output_block(64, n_classes)

    def forward(self, x):
        x1 = self.input(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.output(x)

        return x
