'''
Author: Jiaming-Wang wjmecho@163.com
Date: 2021-02-08 09:19:29
LastEditors: Jiaming-Wang wjmecho@163.com
LastEditTime: 2024-01-15 23:42:53
FilePath: /model/dic.py
Description: lstm版本
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from .base_net import *
from .mybase import *
from torchvision.transforms import *
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        num_channels = 4
        n_resblocks = 8
        num_features = 32
        act_type = 'prelu'
        norm_type = None
        self.num_steps = 2
        n_channels = 32

        self.head = ConvBlock(1, 64, 3, 1, 1, activation='prelu', norm=None, bias = False)
        self.head1 = ConvBlock(num_channels, 64, 3, 1, 1, activation='prelu', norm=None, bias = False)
        self.f1 = ConvBlock(128, 64, 3, 1, 1, activation='prelu', norm=None, bias = False)
        
        self.inc = (DoubleConv(64, 64))
        self.inc1 = (DoubleConv(64, 64))
        self.down1 = (Down(64, 64))
        self.down2 = (Down(64, 64))
        self.down3 = (Down(64, 64))
        self.down4 = (Down1(64, 64))
        self.up1 = (Up(64, 64, False))
        self.up2 = (Up(64, 64, False))
        self.up3 = (Up(64, 64, False))
        self.up4 = (Up(64, 64, False))
        self.outc = (OutConv(64, 4))
        
        self.pslstm11 = pslstm(64)
        
    def forward(self, l_ms, b_ms, x_pan):
        pan_img = x_pan
        x1 = self.head(x_pan)
        x_ms = self.head1(b_ms)
        c_t = torch.ones_like(x1)
        c_t1 = torch.ones_like(x1)
        c = [c_t, c_t1]

        f1 = self.f1(torch.cat([x1, x_ms],1))
        
        x1_pan = self.inc(x1)
        x_ms = self.inc1(x_ms)
        
        c, x2 = self.pslstm11(c, f1, x_ms, x1_pan)
        c, x3, x_ms, x_pan = self.down1(c, x2, x_ms, x1_pan)
        c, x4, x_ms, x_pan = self.down2(c, x3, x_ms, x_pan)
        c, x5, x_ms, x_pan = self.down3(c, x4, x_ms, x_pan)
        c, x6, x_ms, x_pan = self.down4(c, x5, x_ms, x_pan)

        x, _ = self.up1(x6, x5)
        x, _ = self.up2(x, x4)
        x, _ = self.up3(x, x3)
        x, vil = self.up4(x, x2)
        logits = self.outc(x) + b_ms

        # rows, cols = 128, 128
        crow, ccol = logits.shape[2] // 2, logits.shape[3] // 2
        low_pass_filter = torch.tensor(np.ones((logits.shape[0], logits.shape[1], logits.shape[2], logits.shape[3]), np.uint8)).to(torch.device('cuda'))
        low_pass_filter[:,:, crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1
        f=torch.fft.fft2(logits, norm='backward')	#傅里叶变换
        # print(torch.angle(f).shape)
        mag=torch.angle(f)*low_pass_filter
        f=torch.fft.fft2(torch.cat([pan_img, pan_img, pan_img, pan_img], 1), norm='backward')	#将傅里叶变换后的零频率移到中间位置
        # print(f.shape)
        mag1=torch.angle(f)*low_pass_filter
        return logits, [mag, mag1]

class pslstm1(nn.Module):
    def __init__(self, num_features):
        super(pslstm1, self).__init__()
        
        self.f = ConvBlock(num_features*2, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
        self.i = ConvBlock(num_features*3, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
        self.c = ConvBlock(num_features*3, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
        self.o = ConvBlock(num_features*2, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
        
    def forward(self, c, x_t_1, h_t_1, ps_t_1):
        
        c_t_1 = c[0]
        c_t_2 = c[1]
        f_t = torch.sigmoid(self.f(torch.cat([ps_t_1, x_t_1], 1))) ## forget
        i_t = torch.sigmoid(self.i(torch.cat([ps_t_1, x_t_1, h_t_1], 1))) ## save cell statu
        c_hat_t = torch.sigmoid(self.c(torch.cat([ps_t_1, x_t_1, h_t_1], 1))) 
        c_t = f_t * c_t_1 + i_t * c_hat_t ## update cell statu
        o_t = torch.sigmoid(self.o(torch.cat([ps_t_1, x_t_1], 1))) 
        h_t = o_t * torch.tanh(c_t) ## output

        return [c_t, c_t_2], h_t
    
# class pslstm(nn.Module):
#     def __init__(self, num_features):
#         super(pslstm, self).__init__()
#         self.f = ConvBlock(num_features*2, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
#         self.i = ConvBlock(num_features*3, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
#         self.c = ConvBlock(num_features*3, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
#         self.o = ConvBlock(num_features*2, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)

#         self.f_p = ConvBlock(num_features*2, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
#         self.i_p = ConvBlock(num_features*3, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
#         self.c_p = ConvBlock(num_features*3, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
#         self.o_p = ConvBlock(num_features*2, num_features, 3, 1, 1, activation='prelu', norm=None, bias = True)
        
#     def forward(self, c_t_1, c_t_p, x_t_1, h_t_1, ps_t_1):
        
#         x_t_1_corrupted = torch.fft.fftn(x_t_1)
#         h_t_1_corrupted = torch.fft.fftn(h_t_1)
#         ps_t_1_corrupted = torch.fft.fftn(ps_t_1)

#         f_t = torch.sigmoid(self.f(torch.cat([ps_t_1_corrupted.real, x_t_1_corrupted.real], 1))) ## forget
#         i_t = torch.sigmoid(self.i(torch.cat([ps_t_1_corrupted.real, x_t_1_corrupted.real, h_t_1_corrupted.real], 1))) ## save cell statu
#         c_hat_t = torch.sigmoid(self.c(torch.cat([ps_t_1_corrupted.real, x_t_1_corrupted.real, h_t_1_corrupted.real], 1))) 
#         c_t = f_t * c_t_1 + i_t * c_hat_t ## update cell statu
#         o_t = torch.sigmoid(self.o(torch.cat([ps_t_1_corrupted.real, x_t_1_corrupted.real], 1))) 
#         h_t = o_t * torch.tanh(c_t) ## output

#         f_t = torch.sigmoid(self.f(torch.cat([ps_t_1_corrupted.imag, x_t_1_corrupted.imag], 1))) ## forget
#         i_t = torch.sigmoid(self.i(torch.cat([ps_t_1_corrupted.imag, x_t_1_corrupted.imag, h_t_1_corrupted.imag], 1))) ## save cell statu
#         c_hat_t = torch.sigmoid(self.c(torch.cat([ps_t_1_corrupted.imag, x_t_1_corrupted.imag, h_t_1_corrupted.imag], 1))) 
#         c_t_p = f_t * c_t_p + i_t * c_hat_t ## update cell statu
#         o_t = torch.sigmoid(self.o(torch.cat([ps_t_1_corrupted.imag, x_t_1_corrupted.imag], 1))) 
#         h_t_p = o_t * torch.tanh(c_t) ## output
        
#         f_reconstructed = torch.complex(h_t, h_t_p)
#         h_t = torch.fft.ifftn(f_reconstructed)
#         h_t = torch.tensor(h_t, dtype=torch.float32)

#         return c_t, c_t_p, h_t


class pslstm(nn.Module):
    def __init__(self, num_features):
        super(pslstm, self).__init__()
        self.f = ConvBlock(num_features*2, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        self.i = ConvBlock(num_features*3, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        self.c = ConvBlock(num_features*3, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        self.o = ConvBlock(num_features*2, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        
        self.f1 = ConvBlock(num_features*2, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        self.i1 = ConvBlock(num_features*3, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        self.c1 = ConvBlock(num_features*3, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        self.o1 = ConvBlock(num_features*2, num_features, 1, 1, 0, activation='lrelu', norm=None, bias = True)
        
    def forward(self, c, x_t_1, h_t_1, ps_t_1):
        
        c_t_1 = c[0]
        c_t_2 = c[1]

        _, _, H, W = c_t_1.shape
        x_t_1_corrupted = torch.fft.fft2(x_t_1, norm='backward')
        h_t_1_corrupted = torch.fft.fft2(h_t_1, norm='backward')
        ps_t_1_corrupted = torch.fft.fft2(ps_t_1, norm='backward')

        f_t = torch.sigmoid(self.f(torch.cat([torch.angle(ps_t_1_corrupted), torch.angle(x_t_1_corrupted)], 1))) ## forget
        i_t = torch.sigmoid(self.i(torch.cat([torch.angle(ps_t_1_corrupted), torch.angle(x_t_1_corrupted), torch.angle(h_t_1_corrupted)], 1))) ## save cell statu
        c_hat_t = torch.sigmoid(self.c(torch.cat([torch.angle(ps_t_1_corrupted), torch.angle(x_t_1_corrupted), torch.angle(h_t_1_corrupted)], 1))) 

        c_t = f_t * c_t_1 + i_t * c_hat_t ## update cell statu
        c_t = torch.tensor(c_t, dtype=torch.float32)
        o_t = torch.sigmoid(self.o(torch.cat([torch.angle(ps_t_1_corrupted), torch.angle(x_t_1_corrupted)], 1))) 
        h_t = o_t * torch.tanh(c_t) ## output


        f_t = torch.sigmoid(self.f1(torch.cat([torch.abs(ps_t_1_corrupted), torch.abs(x_t_1_corrupted)], 1))) ## forget
        i_t = torch.sigmoid(self.i1(torch.cat([torch.abs(ps_t_1_corrupted), torch.abs(x_t_1_corrupted), torch.abs(h_t_1_corrupted)], 1))) ## save cell statu
        c_hat_t = torch.sigmoid(self.c1(torch.cat([torch.abs(ps_t_1_corrupted), torch.abs(x_t_1_corrupted), torch.abs(h_t_1_corrupted)], 1))) 
        c_t1 = f_t * c_t_2 + i_t * c_hat_t ## update cell statu
        c_t1 = torch.tensor(c_t1, dtype=torch.float32)
        o_t = torch.sigmoid(self.o1(torch.cat([torch.abs(ps_t_1_corrupted), torch.abs(x_t_1_corrupted)], 1))) 
        h_t1 = o_t * torch.tanh(c_t1) ## output
        
        # pha_fus = self.conv(torch.cat([torch.angle(ps_t_1_corrupted), torch.angle(h_t_1_corrupted)], 1))
        real = torch.abs(h_t1) * torch.cos(h_t)+1e-8
        imag = torch.abs(h_t1) * torch.sin(h_t)+1e-8
        out = torch.complex(real, imag)+1e-8

        h_t = torch.abs(torch.fft.ifft2(out, s=(H, W), norm='backward')) + x_t_1

        return [c_t, c_t1], h_t
           
class FBlock(nn.Module):
    def __init__(self,
                 num_features,
                 num_channels):
        super(FBlock, self).__init__()
        act_type = 'prelu'
        norm_type = None
        self.fun1 = ConvBlock(num_features, num_features, 3, 1, 1, activation='prelu', norm=None, bias = False)
        self.fun2 = FeedbackBlockCustom(num_features, 6, 4, act_type, norm_type, num_features)
        self.fun3 = ConvBlock(num_features, num_channels, 3, 1, 1, activation='prelu', norm=None, bias = False)
        
    def forward(self, x):
        x = self.fun1(x)
        x = self.fun2(x)
        x = self.fun3(x)
        return x

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
        self.maxpool_conv = nn.MaxPool2d(2)
        self.lstm = pslstm1(in_channels)
        self.conv = DoubleConv(in_channels, out_channels)
        self.conv1 = DoubleConv(in_channels, out_channels)

    def forward(self, c, x, x_ms, x_pan):
        c_t = self.maxpool_conv(c[0])
        c_t1 = self.maxpool_conv(c[1])
        x = self.maxpool_conv(x)
        x_pan = self.maxpool_conv(x_pan)
        x_pan = self.conv(x_pan)
        x_ms = self.maxpool_conv(x_ms)
        x_ms = self.conv1(x_ms)
        c, x = self.lstm([c_t, c_t1], x, x_ms, x_pan)
        return c, x, x_ms, x_pan

class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)
        self.lstm = pslstm(in_channels)
        self.conv = DoubleConv(in_channels, out_channels)
        self.conv1 = DoubleConv(in_channels, out_channels)

    def forward(self, c, x, x_ms, x_pan):
        c_t = self.maxpool_conv(c[0])
        c_t1 = self.maxpool_conv(c[1])

        x = self.maxpool_conv(x)
        x_pan = self.maxpool_conv(x_pan)
        x_pan = self.conv(x_pan)
        x_ms = self.maxpool_conv(x_ms)
        x_ms = self.conv1(x_ms)
        c, x = self.lstm([c_t, c_t1], x, x_ms, x_pan)
        return c, x, x_ms, x_pan

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(96, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(96, out_channels)
        # self.cat1 = Mct(in_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # x = self.cat(torch.cat([x1, x2], dim=1), x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x), x


class Mct(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels):
        super().__init__()

        self.ca = ChannelAttention(96)
        self.conv = nn.Conv2d(96,32,1,1,0)
        
        # self.ca1 = ChannelAttention(96)
        # self.conv1 = ConvBlock(96, 32, 3, 1, 1, activation='prelu', norm=None, bias = True)

    def forward(self, x1, x2):
        
        x1_corrupted = torch.fft.fft2(x1)
        mag_x1 = torch.abs(x1_corrupted)
        pha_x1 = torch.angle(x1_corrupted)

        x2_1, x2_2 = torch.chunk(x2, 2, dim=1)
        
        x2_corrupted = torch.fft.fftn(x2_1)
        mag_x2 = torch.abs(x2_corrupted)
        pha_x2 = torch.angle(x2_corrupted)
        
        # cat_x = torch.cat([mag_x1, mag_x2], dim=1)
        # cat_x = self.conv(self.ca(cat_x))
        # cat_x = (mag_x1 + mag_x2)/2
        real = mag_x1 * torch.cos(pha_x1)
        imag = mag_x1 * torch.sin(pha_x1)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        #平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        #MLP  除以16是降维系数
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False) #kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        #结果相加
        out = avg_out + max_out
        return self.sigmoid(out)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
         
if __name__ == '__main__':       
    x = torch.randn(1,4,32,32)
    y = torch.randn(1,4,128,128)
    z = torch.randn(1,1,128,128)
    arg = []
    Net = Net(arg)
    out = Net(x, y, z)
    print(out.shape)
    