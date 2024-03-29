import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################
# Classification #
##################

class Classifier_Net(nn.Module):
    def __init__(self, hasHCAL, hiddenLayerNeurons, nHiddenLayers, dropoutProb):
        super().__init__()
        self.hasHCAL = hasHCAL
        if (hasHCAL):
            self.input = nn.Linear(25 * 25 * 25 + 5 * 5 * 60, hiddenLayerNeurons)
        else:
            self.input = nn.Linear(25 * 25 * 25, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.nHiddenLayers = nHiddenLayers
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons, 2)
    def forward(self, x1, x2):
        x1 = x1.view(-1, 25 * 25 * 25)
        if (self.hasHCAL):
            x2 = x2.view(-1, 5 * 5 * 60)
            x = torch.cat([x1, x2], 1)
        else:
            x = x1
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden(x))
            x = self.dropout(x)
        x = F.softmax(self.output(x), dim=1)
        return x

class Classifier():
    def __init__(self, hasHCAL, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate):
        self.net = Classifier_Net(hasHCAL, hiddenLayerNeurons, nHiddenLayers, dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
