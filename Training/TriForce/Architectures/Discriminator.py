import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

##################
# Discriminator #
##################

class Discriminator_Net(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.conv_1 = nn.Conv3d(1, 32, (5, 5, 5), padding=2)
        self.dropout = nn.Dropout(0.5) # should dropout be an optional parameter?
        self.padding1 = nn.ConstantPad3d((2,2,2,2,2,2), 0)
        self.conv_2 = nn.Conv3d(32, 8, (5, 5, 5))
        self.batchnorm_1 = nn.BatchNorm3d(8)
        self.conv_3 = nn.Conv3d(8, 8, (5, 5, 5))
        self.padding2 = nn.ConstantPad3d((1,1,1,1,1,1), 0)
        self.batchnorm_2 = nn.BatchNorm3d(8)
        self.conv_4 = nn.Conv3d(8, 8, (5, 5, 5))
        self.batchnorm_3 = nn.BatchNorm3d(8)
        self.avgpool = nn.AvgPool3d((2, 2, 2))
        self.fake = nn.Linear(10648, 1)
        self.aux = nn.Linear(10648, 1)

    def forward(self, x):
        # Input shape = (1, 25, 25, 25) 
        x = Variable(x.view(-1,1,25,25,25))
        x = F.leaky_relu(self.conv_1(x))
        x = self.dropout(x)
        x = self.padding1(x)
        x = F.leaky_relu(self.conv_2(x))
        x = self.batchnorm_1(x)
        x = self.dropout(x)
        x = self.padding1(x)
        x = F.leaky_relu(self.conv_3(x))
        x = self.batchnorm_2(x)
        x = self.dropout(x)
        x = self.padding2(x)
        x = F.leaky_relu(self.conv_4(x))
        x = self.batchnorm_3(x)
        x = self.dropout(x)
        x = self.avgpool(x)

        flat = x.view(10648)
        
        fake = F.sigmoid(self.fake(flat))
        aux = self.aux(flat) 
        ecal = torch.sum(x) 

        return fake, aux, ecal


class Net():
    def __init__(self, options):
        self.net = Discriminator_Net(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = nn.CrossEntropyLoss() 
