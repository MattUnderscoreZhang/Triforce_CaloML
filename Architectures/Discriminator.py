import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################
# Discriminator #
##################

class Discriminator_Net(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.conv_1 = nn.Conv3d(1, 32, (5, 5, 5), padding=2)
        self.dropout = nn.Dropout(p = options['dropoutProb'])
        self.conv_2 = nn.Conv3d(32, 8, (5, 5, 5), padding=2)
        self.batchnorm_1 = nn.BatchNorm3d(32)
        self.conv_3 = nn.Conv3d(8, 8, (5, 5, 5), padding=2)
        self.batchnorm_2 = nn.BatchNorm3d(32)
        self.conv_4 = nn.Conv3d(8, 8, (5, 5, 5), padding=2)
        self.batchnorm_3 = nn.BatchNorm3d(32)
        self.maxpool = nn.AvgPool3d((2, 2, 2))
        self.fake = nn.Linear(20, 1)
        self.aux = nn.Linear(20, 1)

    def forward(self, data):

        # window slice and energy sums
        ECAL = Variable(data["ECAL"].cuda())
        lowerBound = 26 - int(math.ceil(self.windowSizeECAL/2))
        upperBound = lowerBound + self.windowSizeECAL
        ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
        ECAL = ECAL.contiguous().view(-1, 1, self.windowSizeECAL, self.windowSizeECAL, 25)
        ECAL_sum = torch.sum(ECAL.view(-1, self.windowSizeECAL * self.windowSizeECAL * 25), dim = 1).view(-1, 1) * self.inputScaleSumE

        HCAL = Variable(data["HCAL"].cuda())
        lowerBound = 6 - int(math.ceil(self.windowSizeHCAL/2))
        upperBound = lowerBound + self.windowSizeHCAL
        HCAL = HCAL[:, lowerBound:upperBound, lowerBound:upperBound]
        HCAL = HCAL.contiguous().view(-1, 1, self.windowSizeHCAL, self.windowSizeHCAL, 60)
        HCAL_sum = torch.sum(HCAL.view(-1, self.windowSizeHCAL * self.windowSizeHCAL * 60), dim = 1).view(-1, 1) * self.inputScaleSumE

	# net
        x = ECAL
        x = nn.LeakyReLU(self.conv_1(x))
        x = self.dropout(x)
        x = nn.LeakyReLU(self.conv_2(x))
        x = nn.batchnorm_1(x)
        x = self.dropout(x)
        x = nn.LeakyReLU(self.conv_3(x))
        x = nn.batchnorm_2(x)
        x = self.dropout(x)
        x = nn.LeakyReLU(self.conv_4(x))
        x = nn.batchnorm_3(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x1 = F.sigmoid(self.fake(x))
        # x2 = self.aux(x)
        # return x1, x2
        return x1

class Net():
    def __init__(self, options):
        self.net = Discriminator_Net(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = nn.CrossEntropyLoss()
