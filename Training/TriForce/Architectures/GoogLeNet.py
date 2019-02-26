import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb # NOQA
from Architectures import LossFunctions


##################
# Classification #
##################

epsilon = 1e-07
CLASSIFICATION, REGRESSION = 0, 1


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):

        super().__init__()

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes, eps=epsilon),
            nn.ReLU(True),
        )

    def forward(self, x):

        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):

    def __init__(self, options):

        super().__init__()
        self.windowSizeECAL = options['windowSizeECAL']
        self.windowSizeHCAL = options['windowSizeHCAL']
        self.inputScaleSumE = options['inputScaleSumE']
        self.inputScaleEta = options['inputScaleEta']
        self.inputScalePhi = options['inputScalePhi']
        self.outputs = []
        for particle_class in options['classPdgID']:
            self.outputs += [(str(particle_class)+"_classification", CLASSIFICATION)]
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION), ("phi_regression", REGRESSION)]

        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192, eps=epsilon),
            nn.ReLU(True),
        )

        self.norm = nn.InstanceNorm3d(1)
        # self.norm = nn.BatchNorm3d(1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.dense = nn.Linear(1024 + 4, 1024)  # window size of 25, plus reco angles and energy sums
        self.linear = nn.Linear(1024 + 5, len(self.outputs))  # output layer

    def forward(self, data):

        # ECAL slice and energy sum
        ECAL = Variable(data["ECAL"].cuda())
        lowerBound = math.ceil(ECAL.shape[0]/2) - int(math.ceil(self.windowSizeECAL/2))
        upperBound = lowerBound + self.windowSizeECAL
        ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
        ECAL = ECAL.contiguous().view(-1, 1, self.windowSizeECAL, self.windowSizeECAL, 25)
        # ECAL_sum = torch.sum(ECAL, dim = 1).view(-1, 1) * self.inputScaleSumE
        ECAL_sum = ECAL.sum(2).sum(2).sum(2) * self.inputScaleSumE
        # HCAL slice to get energy sum
        if (self.windowSizeHCAL > 0):
            HCAL = Variable(data["HCAL"].cuda())
            lowerBound = math.ceil(ECAL.shape[0]/2) - int(math.ceil(self.windowSizeHCAL/2))
            upperBound = lowerBound + self.windowSizeHCAL
            HCAL = HCAL[:, lowerBound:upperBound, lowerBound:upperBound]
            HCAL = HCAL.contiguous().view(-1, 1, self.windowSizeHCAL, self.windowSizeHCAL, 60)
            # HCAL_sum = torch.sum(HCAL, dim = 1).view(-1, 1) * self.inputScaleSumE
            HCAL_sum = HCAL.sum(2).sum(2).sum(2) * self.inputScaleSumE
        else:
            HCAL_sum = Variable(torch.zeros(ECAL_sum.size()).cuda())
        # reco angles
        recoEta = Variable(data["recoEta"].cuda()).view(-1, 1) * self.inputScaleEta
        recoPhi = Variable(data["recoPhi"].cuda()).view(-1, 1) * self.inputScaleEta

        # net
        x = self.norm(ECAL)
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # cat angles / energy sums in before dense layer
        x = torch.cat([x, recoPhi, recoEta, ECAL_sum, HCAL_sum], 1)
        x = F.relu(self.dense(x))
        # cat angles / energy sums back in before final layer
        x = torch.cat([x, recoPhi, recoEta, ECAL_sum, HCAL_sum, torch.ones([data['ECAL'].shape[0], 1]).cuda()], 1)
        x = self.linear(x)
        # preparing output
        return_data = {}
        for i, (label, activation) in enumerate(self.outputs):
            if activation == CLASSIFICATION:
                if 'classification' in return_data.keys():
                    return_data['classification'] = torch.stack((return_data['classification'], x[:, i]))
                else:
                    return_data['classification'] = x[:, i]
            else:
                return_data[label] = x[:, i]
        return_data['classification'] = F.softmax(return_data['classification'].transpose(0, 1), dim=1)
        return return_data


class Net():

    def __init__(self, options):

        self.net = GoogLeNet(options)
        # self.net = torch.nn.DataParallel(GoogLeNet(options), device_ids=[0,1,2,3,4,5,6,7,8,9]).cuda()
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        # self.lossFunction = LossFunctions.classificationOnlyLossFunction
        self.lossFunction = LossFunctions.combinedLossFunction
