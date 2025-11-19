#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:48:27
LastEditTime: 2023-03-09 10:48:51
Description: 
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class T_Net(nn.Module):
    def __init__(self, args):
        super(T_Net, self).__init__()

        base_filter = 64
        self.out_channels = args['data']['n_colors']

        self.head = ConvBlock(self.out_channels, 64, 3, 1, 1, activation='relu', norm=None, bias = True)

        res_block = []
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(64, 3, 1, 1, 1, activation='prelu', norm=None))
        self.res_block = nn.Sequential(*res_block)

        self.output_conv = ConvBlock(64, self.out_channels+1, 5, 1, 2, activation='relu', norm=None, bias = True)

    def forward(self, l_ms, b_ms, h_ms, x_pan):

        feature = self.head(h_ms)
        feature = self.res_block(feature)
        feature = self.output_conv(feature)

        R = nn.sigmoid(feature[:,:,0:self.out_channels])
        I = nn.sigmoid(feature[:,:,self.out_channels:self.out_channels+1])
        return R, I

class S_Net(nn.Module):
    def __init__(self, args):
        super(S_Net, self).__init__()

        base_filter = 64
        self.out_channels = args['data']['n_colors']

        self.head = ConvBlock(self.out_channels, 64, 3, 1, 1, activation='relu', norm=None, bias = True)

        res_block = []
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(64, 3, 1, 1, 1, activation='prelu', norm=None))
        self.res_block = nn.Sequential(*res_block)

        self.output_conv = ConvBlock(64, self.out_channels+1, 5, 1, 2, activation='relu', norm=None, bias = True)

        self.unet = UNet(self.out_channels+1, 1)

    def forward(self, l_ms, b_ms, h_ms, x_pan):

        feature = self.head(h_ms)
        feature = self.res_block(feature)
        feature = self.output_conv(feature)

        R_1 = nn.sigmoid(feature[:,:,0:self.out_channels])
        I_1 = nn.sigmoid(feature[:,:,self.out_channels:self.out_channels+1])
        I_2 = self.unet(torch.cat([R_1, I_1]))

        return R_1, I_1, I_2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
