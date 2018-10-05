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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_old(nn.Module):
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
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION), ("phi_regression", REGRESSION)]
        # layers
        self.conv0 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv3d(64, 96, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(96, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 192, 3, stride=1) # window size 25
        # self.conv4 = nn.Conv3d(128, 192, (7, 7, 3), stride=1) # window size 51
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
