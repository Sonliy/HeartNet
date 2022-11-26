import torch
import torch.nn as nn
from torchvision.models import resnet34 as resnet
# from torchvision.models import resnet34 as resnet
from Transformer import deit_small_patch16_224 as deit
# from .DeiT112 import deit_small_patch16_224 as deit
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#   CBAM 模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

# 融合模块
class Fusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(Fusion_block, self).__init__()
        self.cbam_cnn = CBAM(ch_1, ch_1 // r_2, kernel_size=3)
        self.cbam_Transformer = CBAM(ch_2, ch_2 // r_2, kernel_size=3)

        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self,g,x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g*W_x)

        g_in = g
        g = self.cbam_cnn(g)
        g = self.sigmoid(g) * g_in
        # Transformer branch
        x_in = x
        x =  self.cbam_Transformer(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse
#  桥注意力模块
class BA_module(nn.Module):
    def __init__(self, pre_channels, cur_channel, reduction=1):
        super(BA_module, self).__init__()
        self.pre_fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
                nn.Linear(cur_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(cur_channel // reduction, cur_channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.size()
        # xxx = pre_layers[0].view(-1,64)
        # pre_fusions = self.pre_fusions[0](xxx)
        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion).view(b, cur_c, 1, 1)

        return att_weights




# HeartNet网络主结构
class HearNet(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(HearNet, self).__init__()

        self.resnet = resnet()
        # if pretrained:
        # self.resnet.load_state_dict(torch.load('./pretrained/resnet50-19c8e357.pth'))
            # self.resnet.load_state_dict(torch.load('./pretrained/resnet34-333f7ec4.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.conv256 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024,
                            out_channels=256,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU())
        self.conv128 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,
                            out_channels=128,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU())
        self.conv64 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256,
                            out_channels=64,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.transformer = deit(pretrained=pretrained)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.up_c = Fusion_block(ch_1=256, ch_2=384, r_2=4, ch_int=256, ch_out=256, drop_rate=drop_rate/2)

        self.up_c_1_1 = Fusion_block(ch_1=128, ch_2=128, r_2=2, ch_int=128, ch_out=128, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=False)

        self.up_c_2_1 = Fusion_block(ch_1=64, ch_2=64, r_2=1, ch_int=64, ch_out=64, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(128, 64, 64, attn=False)
        self.ba = BA_module([256,128],64)
        self.drop = nn.Dropout2d(drop_rate)
        self.feature_extraction = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b = self.transformer(imgs)
        x_b = torch.transpose(x_b, 1, 2)
        x_b = x_b.view(x_b.shape[0], -1, 7, 7)
        x_b = self.drop(x_b)    # t0   特征图

        x_b_1 = self.up1(x_b)
        x_b_1 = self.drop(x_b_1)  # t1  特征图

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        x_b_2 = self.drop(x_b_2)  # t2  特征图

        # top-down path
        x_u = self.resnet.conv1(imgs)   # x_u=(1,64,96,128)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)     # g0 特征图

        x_u_2 = self.resnet.layer1(x_u)    # x_u_2=(1,64,48,64)
        # x_u_2 = self.conv64(x_u_2)   ##
        x_u_2 = self.drop(x_u_2)    # g2   特征图

        x_u_1 = self.resnet.layer2(x_u_2)   # x_u_1=(1,128,24,32)
         ##
        x_u_1 = self.drop(x_u_1)   # g1 特征图

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u) 
        if x_u.size(1)==1024:
            x_u = self.conv256(x_u)
        # joint path
        x_c = self.up_c(x_u, x_b)       #x_c is the out of Bifusion W/4
        x_c_feature_ex = self.feature_extraction(x_c)
        if x_u_1.size(1) == 512:
            x_u_1 = self.conv128(x_u_1)
        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)    # x_u_1=(1,128,24,32)
        x_c_1_1_feature_ex = self.feature_extraction(x_c_1_1)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)
        if x_u_2.size(1) == 256:
            x_u_2 = self.conv64(x_u_2)
        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)   # x_u_2=(1,64,48,64)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1) # joint predict low supervise here
        residual = x_c_2
        x_c_2_feature_ex = self.feature_extraction(x_c_2)
        # attention
        attention = self.ba([x_c_feature_ex, x_c_1_1_feature_ex],x_c_2_feature_ex)
        x_c_2 = x_c_2*attention
        x_c_2 += residual
        x_c_2 = self.relu(x_c_2)

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear')
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear')         # out of Tranformer
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear')
        return map_x, map_1, map_2
        # return map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


