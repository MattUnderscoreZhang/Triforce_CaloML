import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd.Variable as Variable

#############
# Generator #
#############

class Generator_Net(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.dense_1 = nn.Linear(1024, 64*7*7)
        self.conv_1 = nn.Conv3d(8, 64, (6,6,8), padding=(2,2,3))
        init.uniform(self.conv_1.weight) 
        self.batchnorm_1 = nn.BatchNorm3d(64)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='trilinear')
        
        self.conv_2 = nn.Conv3d(64, 6, (6,5,8))
        init.uniform(self.conv_2.weight)
        self.batchnorm_2 = nn.BatchNorm3d(6)
        self.upsample_2 = nn.Upsample(scale_factor=(2,2,3), mode='trilinear')

        self.conv_3 = nn.Conv3d(6, 6 , (3,3,8))
        init.uniform(self.conv_3.weight)
        self.conv_4 = nn.Conv3d(6, 1 , (2,2,2))
        init.xavier_uniform(self.conv_4.weight)


    def forward(self, data):

        # # window slice and energy sums
        # ECAL = Variable(data["ECAL"].cuda())
        # lowerBound = 26 - int(math.ceil(self.windowSizeECAL/2))
        # upperBound = lowerBound + self.windowSizeECAL
        # ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
        # ECAL = ECAL.contiguous().view(-1, 1, self.windowSizeECAL, self.windowSizeECAL, 25)
        # ECAL_sum = torch.sum(ECAL.view(-1, self.windowSizeECAL * self.windowSizeECAL * 25), dim = 1).view(-1, 1) * self.inputScaleSumE

        # HCAL = Variable(data["HCAL"].cuda())
        # lowerBound = 6 - int(math.ceil(self.windowSizeHCAL/2))
        # upperBound = lowerBound + self.windowSizeHCAL
        # HCAL = HCAL[:, lowerBound:upperBound, lowerBound:upperBound]
        # HCAL = HCAL.contiguous().view(-1, 1, self.windowSizeHCAL, self.windowSizeHCAL, 60)
        # HCAL_sum = torch.sum(HCAL.view(-1, self.windowSizeHCAL * self.windowSizeHCAL * 60), dim = 1).view(-1, 1) * self.inputScaleSumE

        # net 
        # should not take ECAL, just noise. 
        # x = ECAL
        # x = (1024, )

        x = torch.rand((1024,)) #place holder. shoud be noise sampled from a normal distrirbution. 
        x = Variable(x) # place holder.  
        x = self.dense_1(x)
        x = x.view(-1,8,7,7,8)
        x = F.leaky_relu(self.conv_1(x)) 
        x = self.batchnorm_1(x)
        x = self.upsample_1(x)
        x = F.pad(x, (1,1,1,1,1,1))
        x = F.pad(x, (0,0,2,2,2,2))
        x = F.leaky_relu(self.conv_2(x))
        x = self.upsample_2(x)
        x = F.pad(x, (3,3,0,0,1,1))
        x = F.leaky_relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        return x

class Net():
    def __init__(self, options):
        self.net = Generator_Net(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = nn.CrossEntropyLoss()
