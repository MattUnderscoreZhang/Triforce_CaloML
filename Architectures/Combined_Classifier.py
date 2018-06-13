import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

##################
# Classification #
##################

class Classifier_Net(nn.Module):
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize):
        super().__init__()
        self.windowSize = windowSize
        self.nHiddenLayers = nHiddenLayers
        self.input = nn.Linear(windowSize * windowSize * 25, hiddenLayerNeurons)
        self.hidden = [None] * self.nHiddenLayers
        self.dropout = [None] * self.nHiddenLayers
        for i in range(self.nHiddenLayers):
            self.hidden[i] = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
            self.hidden[i].cuda()
            self.dropout[i] = nn.Dropout(p = dropoutProb)
            self.dropout[i].cuda()
        self.output = nn.Linear(hiddenLayerNeurons, 2)
    def forward(self, x, _):
        lowerBound = 26 - int(math.ceil(self.windowSize/2))
        upperBound = lowerBound + self.windowSize
        x = x[:, lowerBound:upperBound, lowerBound:upperBound]
        x = x.contiguous().view(-1, self.windowSize * self.windowSize * 25)
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden[i](x))
            x = self.dropout[i](x)
        x = F.softmax(self.output(x), dim=1)
        return x

class Net():
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate, windowSize):
        self.net = Classifier_Net(hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
