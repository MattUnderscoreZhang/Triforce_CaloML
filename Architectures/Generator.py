import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#############
# Generator #
#############

class Generator_Net(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.dense_1 = nn.Linear(1024, 8*8*8)
        self.conv_1 = nn.Conv3d(64, 7, (6, 6, 6), padding=1)
        self.batchnorm_1 = nn.BatchNorm3d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')


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
        x = self.dense_1(x)
        x = x.view(-1, 1, 8, 8, 8)
        x = nn.LeakyReLU(self.conv_1(x))
        x = self.batchnorm_1(x)
        x = self.upsample(x)


        # ZeroPadding3D((2, 2, 0)),
        # Conv3D(6, 6, 5, 8, init='he_uniform'),
        # LeakyReLU(),
        # BatchNormalization(),
        # UpSampling3D(size=(2, 2, 3)),
        # ZeroPadding3D((1,0,3)),
        # Conv3D(6, 3, 3, 8, init='he_uniform'),
        # LeakyReLU(),
        # Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal'),
        # Activation('relu')

        return x

class Net():
    def __init__(self, options):
        self.net = Generator_Net(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = nn.CrossEntropyLoss()
