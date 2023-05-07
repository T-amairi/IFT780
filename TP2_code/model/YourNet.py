# -*- coding:utf-8 -*- 

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from model.CNNBaseModel import CNNBaseModel
from layers.CNNBlocks import ResidualBlock, DenseBlock, BottleneckBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourNet.  Le réseau est constitué de

    1) quelques blocs d'opérations de base du type «conv-batch-norm-relu»
    2) 1 (ou plus) bloc dense inspiré du modèle «denseNet)
    3) 1 (ou plus) bloc résiduel inspiré de «resNet»
    4) 1 (ou plus) bloc de couches «bottleneck» avec ou sans connexion résiduelle, c’est au choix
    5) 1 (ou plus) couches pleinement connectées 
    
    NOTE : le code des blocks résiduel, dense et bottleneck doivent être mis dans le fichier CNNBlocks.py
    Aussi, vous pouvez vous inspirer du code de CNNVanilla.py pour celui de *YourNet*

'''

class YourNet(CNNBaseModel):

    def __init__(self, num_classes=10):
        super(YourNet, self).__init__()

        #############################
        # Conv-BatchNorm-Relu layer #
        #############################
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        ###############
        # Dense Layer #
        ###############
        bottleneck_size, growth_rate = 2, 8
        self.dense_layer1 = self.getDenseLayer(3, 64, bottleneck_size, growth_rate)
        in_hidden = 64 + 3 * growth_rate

        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_hidden, (in_hidden) // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        in_hidden = in_hidden // 2
        self.dense_layer2 = self.getDenseLayer(3, in_hidden, bottleneck_size, growth_rate)
        in_hidden = in_hidden + 3 * growth_rate

        ##################
        # Residual layer #
        ##################
        self.residual_layer = self.getResidualLayer(2, in_hidden, 68)

        ####################
        # Bottleneck layer #
        ####################
        self.bottleneck_layer = BottleneckBlock(68, 0.5)

        #########################
        # Fully connected layer #
        #########################
        self.fc_layer = nn.Sequential(
            nn.Linear(34 * 6 * 6, num_classes)
        )

    def getDenseLayer(self, num_layers, in_channels, bottleneck_size, growth_rate):
        layers = list()

        for layer_idx in range(num_layers):
            layers.append(DenseBlock(in_channels + layer_idx * growth_rate, bottleneck_size, growth_rate))

        return nn.Sequential(*layers)
    
    def getResidualLayer(self, num_layers, in_channels, out_channels):
        layers = list()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                layers.append(ResidualBlock(in_channels, out_channels))
            else:
                layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.dense_layer1(out)
        out = self.transition_layer(out)
        out = self.dense_layer2(out)
        out = self.residual_layer(out)
        out = self.bottleneck_layer(out)
        return self.fc_layer(out.view(out.size(0), -1))
    
'''
FIN DE VOTRE CODE
'''
