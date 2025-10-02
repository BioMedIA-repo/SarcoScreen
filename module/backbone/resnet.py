import torch.nn as nn
import torch
import torchvision.models as models
from collections import OrderedDict
from collections import namedtuple

res = {
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet152': models.resnet152,
}

class ResNet(nn.Module):
    def __init__(self, backbone, pretrained=True, channels=3):
        super(ResNet, self).__init__()
        resnet = res[backbone](pretrained=pretrained)  # pretrained ImageNet

        if channels == 4:
            conv1_weight = resnet.conv1.weight.data
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1.weight.data = torch.cat((conv1_weight, conv1_weight.mean(dim=1, keepdim=True)), dim=1)

        self.topconvs = nn.Sequential(
            OrderedDict(list(resnet.named_children())[0:3]))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.topconvs(x)
        layer0 = x
        x = self.max_pool(x)
        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        layer2 = x
        x = self.layer3(x)
        layer3 = x
        x = self.layer4(x)
        layer4 = x
        res_outputs = namedtuple("SideOutputs", ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'])
        out = res_outputs(layer0=layer0, layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
        return out

# class ResNet(nn.Module):
#     def __init__(self, backbone, pretrained=True, channels = 3):
#         super(ResNet, self).__init__()
#         resnet = res[backbone](pretrained=pretrained)  # pretrained ImageNet
#
#         self.nlb4 = None
#
#         if channels==4:
#             conv1_weight = resnet.conv1.weight.data
#             resnet.conv1 = nn.Conv2d(4, 64,kernel_size=7, stride=2, padding=3,bias=False)
#             resnet.conv1.weight.data = torch.cat((conv1_weight, conv1_weight.mean(dim=1, keepdim=True)), dim=1)
#             self.nlb4 = NLBlockND(in_channels=256, inter_channels=None, mode='concatenate', dimension=2, bn_layer=True)
#         #     self.nlb0 = NLBlockND(in_channels=4, inter_channels=None, mode='concatenate', dimension=2, bn_layer=True)
#         # else:
#         #     self.nlb0 = NLBlockND(in_channels=3,inter_channels=None,mode='concatenate',dimension=2,bn_layer=True)
#
#         self.topconvs = nn.Sequential(
#             OrderedDict(list(resnet.named_children())[0:3]))
#
#         # self.nlb1 = NLBlockND(in_channels=64, inter_channels=None, mode='concatenate', dimension=2, bn_layer=True)
#
#         # 通道和空间注意力
#         # self.ca0 = ChannelAttention(channels = 64)
#         # self.sa0 = SpatialAttention()
#
#         #SENet注意力机制
#         # self.se0 = SqueezeExcitation(channels = 64)
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = resnet.layer1
#         # ResNet18、ResNet34
#         self.ar1 = ARL(in_channels=64, out_channels=64, layer=1, weight=0.001)
#
#         # ResNet50
#         # self.ar1 = ARL(in_channels=64, out_channels=256, layer=1, weight=0.01)
#         # self.ca1 = ChannelAttention(channels = 64)
#         # self.sa1 = SpatialAttention()
#         # self.se1 = SqueezeExcitation(channels = 64)
#
#         # self.nlb2 = NLBlockND(in_channels=64, inter_channels=None, mode='concatenate', dimension=2, bn_layer=True)
#
#         self.layer2 = resnet.layer2
#         self.ar2 = ARL(in_channels=64, out_channels=128, layer=2, weight=0.01)
#         # self.ar2 = ARL(in_channels=256, out_channels=512, weight=0.1)
#         # self.ca2 = ChannelAttention(channels = 128)
#         # self.sa2 = SpatialAttention()
#         # self.se2 = SqueezeExcitation(channels = 128)
#
#         # self.nlb3 = NLBlockND(in_channels=128, inter_channels=None, mode='concatenate', dimension=2, bn_layer=True)
#
#         self.layer3 = resnet.layer3
#         self.ar3 = ARL(in_channels=128, out_channels=256, layer=3, weight=0.1)
#         # self.ar3 = ARL(in_channels=512, out_channels=1024, weight=0.5)
#         # self.ca3 = ChannelAttention(channels = 256)
#         # self.sa3 = SpatialAttention()
#         # self.se3 = SqueezeExcitation(channels = 256)
#
#         self.nlb4 = NLBlockND(in_channels=256, inter_channels=None, mode='concatenate', dimension=2, bn_layer=True)
#         # self.nlb4 = NLBlockND(in_channels=256, inter_channels=None, mode='embedded', dimension=2, bn_layer=True)
#
#         self.layer4 = resnet.layer4
#         self.ar4 = ARL(in_channels=256, out_channels=512, layer=4, weight = 1.0)
#         # self.ar4 = ARL(in_channels=1024, out_channels=2048, weight = 1.0)
#         # self.ca4 = ChannelAttention(channels = 512)
#         # self.sa4 = SpatialAttention()
#         # self.se4 = SqueezeExcitation(channels=512)
#
#     def forward(self, x):
#         # x = self.nlb0(x)
#         x = self.topconvs(x)
#         layer0 = x
#
#         # x = self.ca0(x) * x
#         # x = self.sa0(x) * x
#         # CBMA0 = x
#
#         # x = self.se0(x)
#         # SE0 = x
#         # x = self.nlb1(x)
#         x = self.max_pool(x)
#
#         identity_1 = x
#         x = self.layer1(x)
#         layer1 = x
#         x = self.ar1(layer1, identity_1)
#         ARL1 = x
#
#         # x = self.ca1(x) * x
#         # x = self.sa1(x) * x
#         # CBMA1 = x
#         # x = self.se1(x)
#         # SE1 = x
#
#         # x = self.nlb2(x)
#         identity_2 = x
#         x = self.layer2(x)
#         layer2 = x
#         x = self.ar2(layer2, identity_2)
#         ARL2 = x
#
#         # x = self.ca2(x) * x
#         # x = self.sa2(x) * x
#         # CBMA2 = x
#         # x = self.se2(x)
#         # SE2 = x
#
#         # x = self.nlb3(x)
#         identity_3 = x
#         x = self.layer3(x)
#         layer3 = x
#         x = self.ar3(layer3, identity_3)
#         ARL3 = x
#
#         # x = self.ca3(x) * x
#         # x = self.sa3(x) * x
#         # CBMA3 = x
#         # x = self.se3(x)
#         # SE3 = x
#
#         if self.nlb4 is not None:
#             x = self.nlb4(x)
#         identity_4 = x
#         x = self.layer4(x)
#         layer4 = x
#         x = self.ar4(layer4, identity_4)
#         ARL4 = x
#
#         # x = self.ca4(x) * x
#         # x = self.sa4(x) * x
#         # CBMA4 = x
#         # x = self.se4(x)
#         # SE4 = x
#
#         res_outputs = namedtuple("SideOutputs",
#                                  ['layer0',
#                                   'layer1', 'ARL1',
#                                   'layer2', 'ARL2',
#                                   'layer3', 'ARL3',
#                                   'layer4', 'ARL4'])
#         out = res_outputs(layer0=layer0,
#                           layer1=layer1, ARL1=ARL1,
#                           layer2=layer2, ARL2=ARL2,
#                           layer3=layer3, ARL3=ARL3,
#                           layer4=layer4, ARL4=ARL4)
#
#         # res_outputs = namedtuple("SideOutputs",
#         #                          ['layer0', 'CBMA0',
#         #                           'layer1', 'CBMA1',
#         #                           'layer2', 'CBMA2',
#         #                           'layer3', 'CBMA3',
#         #                           'layer4', 'CBMA4'])
#         # out = res_outputs(layer0=layer0, CBMA0=CBMA0,
#         #                   layer1=layer1, CBMA1=CBMA1,
#         #                   layer2=layer2, CBMA2=CBMA2,
#         #                   layer3=layer3, CBMA3=CBMA3,
#         #                   layer4=layer4, CBMA4=CBMA4)
#
#         # res_outputs = namedtuple("SideOutputs",
#         #                          ['layer0', 'SE0',
#         #                           'layer1', 'SE1',
#         #                           'layer2', 'SE2',
#         #                           'layer3', 'SE3',
#         #                           'layer4', 'SE4'])
#         # out = res_outputs(layer0=layer0, SE0=SE0,
#         #                   layer1=layer1, SE1=SE1,
#         #                   layer2=layer2, SE2=SE2,
#         #                   layer3=layer3, SE3=SE3,
#         #                   layer4=layer4, SE4=SE4)
#
#         # res_outputs = namedtuple("SideOutputs", ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'])
#         # out = res_outputs(layer0=layer0, layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
#         return out

# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(channels // ratio, channels, 1, bias=False)

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

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# SENet注意力机制
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // ratio, channels, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Attention Residual Learning
class ARL(nn.Module):
    def __init__(self, in_channels, out_channels, layer = 0, weight = 0.001):
        super(ARL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = layer
        self.weight = weight
        self.w = nn.Parameter(torch.Tensor([self.weight]))
        # self.w = nn.Parameter(torch.Tensor([0.01])) # default requires_grad=True
        self.spatital = nn.Softmax2d()
        self.relu = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, residual, identity):
        # self.identity = identity
        # self.residual = residual
        if self.layer == 1:
            identity = self.conv1x1(identity)
        else:
            identity = self.conv3x3(identity)
        # spatitial_attention = self.spatital(self.residual)
        # temp = self.weight * spatitial_attention * self.identity
        # temp1 = self.residual + temp
        # return self.relu(temp1)
        # return self.relu(residual + self.weight * self.spatital(residual) * identity)
        # return self.relu(identity + residual + self.w * self.spatital(residual) * identity)
        return self.relu(residual + self.w * self.spatital(residual) * identity)


if __name__ == '__main__':
    RN = ResNet('resnet18', False)
    print(RN)