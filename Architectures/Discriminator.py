import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################
# Discriminator #
##################

class Classifier_Net(nn.Module):
    def __init__(self, hasHCAL, hiddenLayerNeurons, nHiddenLayers, dropoutProb):
        super().__init__()
        self.conv_1 = nn.Conv3d(1, 32, (5, 5, 5), padding=2)
        self.dropout = nn.Dropout(p = dropoutProb)
        self.conv_2 = nn.Conv3d(32, 8, (5, 5, 5), padding=2)
        self.batchnorm_1 = nn.BatchNorm3d(32)
        self.conv_3 = nn.Conv3d(8, 8, (5, 5, 5), padding=2)
        self.batchnorm_2 = nn.BatchNorm3d(32)
        self.conv_4 = nn.Conv3d(8, 8, (5, 5, 5), padding=2)
        self.batchnorm_3 = nn.BatchNorm3d(32)
        self.maxpool = nn.AveragePool3d((2, 2, 2))
        self.fake = nn.Linear(20, 1)
        self.aux = nn.Linear(20, 1)

    def forward(self, x1, x2):
        x1 = x1.view(-1, 25 * 25 * 25)
        if (self.hasHCAL):
            x2 = x2.view(-1, 5 * 5 * 60)
            x = torch.cat([x1, x2], 1)
        else:
            x = x1
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

class Classifier():
    def __init__(self, hasHCAL, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate):
        self.net = Classifier_Net(hasHCAL, hiddenLayerNeurons, nHiddenLayers, dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
