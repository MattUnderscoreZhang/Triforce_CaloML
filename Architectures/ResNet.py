import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, pdb
from Architectures import LossFunctions
import numpy as np

##################
# Classification #
##################

CLASSIFICATION, REGRESSION = 0, 1

class ResBlock(nn.Module):
    def __init__(self, NumChannels): 
        super().__init__()
        self.conv0 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
        self.selu0 = nn.SELU()
        self.selu1 = nn.SELU()
    def forward(self, x):
        y = self.conv0(x)
        y = self.selu0(y)
        y = self.conv1(y)
        return self.selu1(torch.add(y, x))

class ResNet(nn.Module):
    def build_layer(self, NumLayers, NumChannels):
        layers = []
        for _ in range(NumLayers):
            layers.append(ResBlock(NumChannels))
        return nn.Sequential(*layers)
    def __init__(self, options):
        super().__init__()
        # window slicing and outputs
        self.windowSizeECAL = options['windowSizeECAL']
        self.outputs = []
        for particle_class in options['classPdgID']:
            self.outputs += [(str(particle_class)+"_classification", CLASSIFICATION)]
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION)]
        # layers
        self.conv0 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv3d(64, 96, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(96, 128, 3, stride=2, padding=1)
        # self.conv4 = nn.Conv3d(128, 192, 3, stride=1) # window size 25
        self.conv4 = nn.Conv3d(128, 192, (7, 7, 3), stride=1) # window size 51
        self.fc1 = nn.Linear(192, 192)
        self.fc2 = nn.Linear(192, 2)
        self.selu0 = nn.SELU()
        self.selu1 = nn.SELU()
        self.selu2 = nn.SELU()
        self.selu3 = nn.SELU()
        self.selu4 = nn.SELU()
        self.selu5 = nn.SELU()
        self.selu6 = nn.SELU()
        self.block0 = self.build_layer(4, 32)
        self.block1 = self.build_layer(4, 64)
        self.block2 = self.build_layer(4, 96)
    def forward(self, data):
        # ECAL slice
        ECAL = Variable(data["ECAL"].cuda())
        lowerBound = 26 - int(math.ceil(self.windowSizeECAL/2))
        upperBound = lowerBound + self.windowSizeECAL
        ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
        x = ECAL.contiguous().view(-1, 1, self.windowSizeECAL, self.windowSizeECAL, 25)
        # net
        x = self.selu0(x)
        x = self.conv0(x)
        x = self.selu1(x)
        x = self.block0(x)
        x = self.conv1(x)
        x = self.selu2(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.selu3(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.selu4(x)
        x = self.conv4(x)
        x = self.selu5(x)
        x = x.view(-1, 192)
        x = self.fc1(x)
        x = self.selu6(x)
        x = self.fc2(x)
        # preparing output
        return_data = {}
        for i, (label, activation) in enumerate(self.outputs):
            if activation == CLASSIFICATION:
                if 'classification' in return_data.keys():
                    return_data['classification'] = torch.stack((return_data['classification'], x[:, i].type(torch.DoubleTensor)))
                else:
                    return_data['classification'] = x[:, i].type(torch.DoubleTensor)
            else:
                return_data[label] = Variable(torch.from_numpy(np.array([0]*ECAL.size(0))).type(torch.FloatTensor)).cuda() # no regression
        return_data['classification'] = F.softmax(return_data['classification'].transpose(0, 1), dim=1).cuda()
        return return_data

class Net():
    def __init__(self, options):
        self.net = ResNet(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = LossFunctions.classificationOnlyLossFunction
