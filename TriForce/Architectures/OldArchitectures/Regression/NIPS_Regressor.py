import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############
# Regression #
##############

class Regressor_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 3, (4, 4, 4))
        self.conv2 = nn.Conv3d(1, 10, (2, 2, 6))
        self.linear1 = nn.Linear(5073, 1000)
        self.linear2 = nn.Linear(1000, 1)
    def forward(self, ECALs, HCALs):
        # ECAL input
        r = ECALs.view(-1, 1, 25, 25, 25)
        model1 = self.conv1(r)
        model1 = F.relu(model1)
        model1 = nn.MaxPool3d(2)(model1)
        model1 = model1.view(model1.size(0), -1)
        # HCAL input
        r = HCALs.view(-1, 1, 5, 5, 60)
        model2 = self.conv2(r)
        model2 = F.relu(model2)
        model2 = nn.MaxPool3d(2)(model2)
        model2 = model2.view(model2.size(0), -1)
        # join the two input models
        bmodel = torch.cat((model1, model2), 1)  # branched model
        # fully connected ending
        bmodel = self.linear1(bmodel)
        bmodel = F.relu(bmodel)
        bmodel = nn.Dropout(p=0.5)(bmodel)
        bmodel = self.linear2(bmodel)
        return bmodel

class Regressor():
    def __init__(self, learningRate, decayRate):
        self.net = Regressor_Net()
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.MSELoss()
