
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from model.psrt import Block
from utils.utils import make_loss

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        base_filter = 64

        out_channels = args['data']['hsi_colors']
        num_channels = out_channels + args['data']['msi_colors']
        
        self.args = args
        dict_channels = 30
        self.dict_channels = dict_channels

        self.f_block = ConvBlock(out_channels, base_filter, 5, 1, 2, activation='relu', norm=None, bias = True)
        self.b1 = DictBlock()
        self.b2 = DictBlock()
        self.b3 = DictBlock()

        self.loss = make_loss(self.args['schedule']['loss'])
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                # torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                # torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, ms_image, lrhs_image):

        bic= F.interpolate(lrhs_image, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        
        fus_img1 = self.b1(bic, bic, ms_image)
        fus_img2 = self.b2(fus_img1, bic, ms_image)
        fus_img3 = self.b3(fus_img2, bic, ms_image)

        return fus_img3
    
    def model_train(self, ms_image, lrhs_image, hs_image):

        out = self.forward(ms_image, lrhs_image)
        loss = self.loss(hs_image, out) / (self.args['data']['batch_size'] * 2)

        return out, loss

class DictBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self):
        super().__init__()

        dict_channels = 30
        self.double_conv = Block(out_num=1, inside_num=1, img_size=64, in_chans=50, embed_dim=32, head=8,
                       win_size=8)
        self.double_conv_1 = nn.Sequential(
            nn.Conv2d(32, dict_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.double_conv1 = Block(out_num=1, inside_num=1, img_size=64, in_chans=dict_channels+4, embed_dim=32, head=8,
                       win_size=8)
        # Block(out_num=1, inside_num=1, img_size=64, in_chans=dict_channels+4, embed_dim=32, head=8,
                    #    win_size=8)
        self.double_conv1_1 = nn.Sequential(
            nn.Conv2d(32, dict_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.refine = Refine(30)

        self.G_Dict = G_Dict()

    def forward(self, feature_lr, bic, img_ms):
        _, _, H, W = feature_lr.shape
        
        dict_lr = self.double_conv(H,W,feature_lr)
        dict_lr = self.double_conv_1(dict_lr)
        lambda_ = self.G_Dict(dict_lr, bic)
        lambda_ = self.refine(lambda_)
        dict_hr = self.double_conv1(H,W,torch.cat([dict_lr, img_ms], 1))
        dict_hr = self.double_conv1_1(dict_hr) + dict_lr
        lambda_ = lambda_.permute(0, 2, 1)
        fus_img = torch.einsum('biwh,bji->bjwh', dict_hr, lambda_)

        return fus_img

class Refine(nn.Module):

    def __init__(self,n_feat):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv1d(n_feat, n_feat, kernel_size=3, stride=1, padding=1)
        self.process = Attention1D(n_feat, n_feat,)
        self.conv_last = nn.Conv1d(n_feat, n_feat, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)+x
        out = self.conv_last(out) 

        return out
    
class Attention1D(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(Attention1D, self).__init__()
        
        # Linear layers to learn the query, key, and value
        self.query_fc = nn.Linear(in_channels, hidden_size)
        self.key_fc = nn.Linear(in_channels, hidden_size)
        self.value_fc = nn.Linear(in_channels, hidden_size)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size]))

    def forward(self, x):
        """
        :param x: input feature tensor (batch_size, channels, sequence_length)
        :return: attention-weighted output, attention scores
        """
        # Transpose to (batch_size, sequence_length, channels) for linear layers
        x = x.transpose(1, 2)

        # Compute query, key, and value matrices
        queries = self.query_fc(x).cuda()
        keys = self.key_fc(x).cuda()
        values = self.value_fc(x).cuda()

        # Attention score matrix (batch_size, sequence_length, sequence_length)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale.cuda()
        
        # Apply softmax to normalize the attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the attention-weighted output
        attention_output = torch.bmm(attention_weights, values)
        
        # Transpose back to (batch_size, channels, sequence_length)
        attention_output = attention_output.transpose(1, 2)
        
        return attention_output
    

class G_Dict(nn.Module):

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01, requires_grad=True))

    def forward(self, dict_l, bic):
        B, C, H, W = dict_l.shape
        dict_l = dict_l.reshape(B, -1, H*W)

        bic_t = bic.reshape(B, -1, H*W)

        DtD = torch.matmul(dict_l, dict_l.permute(0, 2, 1))
        I = torch.eye(30).cuda().unsqueeze(0).expand_as(DtD)

        reg_term = self.alpha * I + DtD
        
        Dtf = torch.bmm(dict_l, bic_t.permute(0, 2, 1))
        lambda_ = torch.linalg.solve(reg_term, Dtf) 
        return lambda_
    
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.inv_conv = InvBlockExp(in_channels, in_channels/2)
    def forward(self, x):
        x = self.inv_conv(x)
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        return self.conv(x1)


class InvBlockExp(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = DenseBlock(self.split_len2, self.split_len1)
        self.G = DenseBlock(self.split_len1, self.split_len2)
        self.H = DenseBlock(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]
    
import torch.nn.init as init
def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

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
        self.down4 = (Down(512, n_classes))
        self.up1 = (Up(n_classes, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return x5, logits
        
if __name__ == '__main__':       
    x = torch.randn(1,120,4,4)
    y = torch.randn(1,120,16,16)
    arg = []
    Net = SVDNet(arg)
    out = Net(x, y)
    # print(out)