# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch 
import torch.nn as nn 
from models.CNNBaseModel import CNNBaseModel 
import torch.nn.functional as F 
from models.CNNBlocks import ResBottBlock

class yourSegNet(CNNBaseModel): 
    """
    Class that implements a brand new segmentation CNN
    """ 
    def __init__(self, num_classes=4, init_weights=True): 
        """
        Builds UNet  model.
        Args:
            num_classes(int): number of classes. default 4 (acdc)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """ 
        super().__init__(num_classes, init_weights) # Couches de base

        self.conv1 = nn.Sequential(ResBottBlock(1, 64), 
                                   ResBottBlock(64, 64), 
                                   ResBottBlock(64, 128),  
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv2 = nn.Sequential(ResBottBlock(128, 256), 
                                   ResBottBlock(256, 256), 
                                   ResBottBlock(256, 512),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv3 = nn.Sequential(ResBottBlock(512, 512), 
                                   ResBottBlock(512, 512))
        
        self.decoder = nn.Sequential(
            ASPP(512, 512),
            ResBottBlock(512, 256),
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(scale_factor=4, mode='bilinear'))

    def forward(self, x): 
        encode_block1 = self.conv1(x) 
        encode_block2 = self.conv2(encode_block1) 
        encode_block3 = self.conv3(encode_block2) 
        decode = self.decoder(encode_block3) 
        return decode

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()))

        for dilatation in dilations :
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilatation, dilation=dilatation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))

        res = torch.cat(_res, dim=1)
        return self.project(res)