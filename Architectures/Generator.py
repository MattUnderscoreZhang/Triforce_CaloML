import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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


    def forward(self, x):
        # Here x should be sapmled from a random distribution.  
        x = Variable(x)   
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