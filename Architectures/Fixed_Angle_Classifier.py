import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from Architectures import LossFunctions

import pdb

##################
# Classification #
##################

class Classifier_Net(nn.Module):
    def __init__(self, options): # hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize):
        super().__init__()
        self.windowSize = options['windowSize']
        self.nHiddenLayers = options['nHiddenLayers']
        self.input = nn.Linear(self.windowSize * self.windowSize * 25, options['hiddenLayerNeurons'])
        self.hidden = [None] * self.nHiddenLayers
        self.dropout = [None] * self.nHiddenLayers
        for i in range(self.nHiddenLayers):
            self.hidden[i] = nn.Linear(options['hiddenLayerNeurons'], options['hiddenLayerNeurons'])
            self.hidden[i].cuda()
            self.dropout[i] = nn.Dropout(p = options['dropoutProb'])
            self.dropout[i].cuda()
        self.output = nn.Linear(options['hiddenLayerNeurons'], 2)
    def forward(self, x):
        x = Variable(x['ECAL'].cuda())
        lowerBound = 26 - int(math.ceil(self.windowSize/2))
        upperBound = lowerBound + self.windowSize
        x = x[:, lowerBound:upperBound, lowerBound:upperBound]
        x = x.contiguous().view(-1, self.windowSize * self.windowSize * 25)
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden[i](x))
            x = self.dropout[i](x)
        x = self.output(x)

        return_data = {}
        return_data['classification'] = F.softmax(x, dim=1)
        # return_data['classification'] = F.softmax(x.transpose(0, 1), dim=1)
        return return_data

class Classifier():
    def __init__(self, options): # hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate, windowSize):
        self.net = Classifier_Net(options) # hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = LossFunctions.classificationOnlyLossFunction
