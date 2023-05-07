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
import torch.nn.functional as F


class SimpleCNNBlock(nn.Module):
    """
    this block is an example of a simple conv-relu-conv-relu block
    with 3x3 convolutions
    """

    def __init__(self, in_channels):
        super(SimpleCNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        return output


""" 
TODO

Suivant l'example ci-haut, vous devez rédiger les classes permettant de créer des :

1- Un bloc résiduel
2- Un bloc dense
3- Un bloc Bottleneck

Ces blocks seront utilisés dans le fichier YouNET.py
"""


class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an 
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        out = self.conv1(x)
        H = self.conv2(out)
        return F.relu(x + H)
     
class DenseBlock(nn.Module):
    """
    This block is the building block of the Dense network. It takes an
    input with in_channels, applies some blocks of convolutional, batchnorm layers
    and then concatenate the output with the original input
    """

    def __init__(self, in_channels, bottleneck_size, growth_rate):
        super(DenseBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bottleneck_size * growth_rate, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(bottleneck_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_size * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False))

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([out, x], dim=1)


class BottleneckBlock(nn.Module):
    """
    This block takes an input with in_channels reduces number of channels by a certain
    parameter "downsample" through kernels of size 1x1, 3x3, 1x1 respectively.
    """

    def __init__(self, in_channels, downsample):
        super(BottleneckBlock, self).__init__()

        out_channels = int(in_channels * downsample)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels ,  kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)) 
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,  in_channels, kernel_size = 3, stride = 1, padding = 0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out) 
        return self.conv3(out)
        