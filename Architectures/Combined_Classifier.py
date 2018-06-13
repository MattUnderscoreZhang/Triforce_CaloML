import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, pdb

##################
# Classification #
##################

CLASSIFICATION, REGRESSION = 0, 1

class Classifier_Net(nn.Module):

    def __init__(self, classes, hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize):
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
        self.outputs = []
        for particle_class in classes:
            self.outputs += [(str(particle_class)+"_classification", CLASSIFICATION)]
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION)]
        self.finalLayer = nn.Linear(hiddenLayerNeurons, len(self.outputs)) # nClasses = 2 for binary classifier

    def forward(self, data):
        x = Variable(data["ECAL"].cuda())
        lowerBound = 26 - int(math.ceil(self.windowSize/2))
        upperBound = lowerBound + self.windowSize
        x = x[:, lowerBound:upperBound, lowerBound:upperBound]
        x = x.contiguous().view(-1, self.windowSize * self.windowSize * 25)
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden[i](x))
            x = self.dropout[i](x)
        x = self.finalLayer(x)
        return_data = {}
        for i, (label, activation) in enumerate(self.outputs):
            if activation == CLASSIFICATION:
                if 'classification' in return_data.keys():
                    return_data['classification'] = torch.stack((return_data['classification'], x[:, i]))
                else:
                    return_data['classification'] = x[:, i]
            else:
                return_data[label] = x[:, i]
        return_data['classification'] = F.softmax(return_data['classification'].transpose(0, 1), dim=1)
        return return_data

# def lossFunction(output, data):
    # truth = Variable(data["pdgID"].cuda())
    # return nn.CrossEntropyLoss(output, truth).data[0]

class Net():
    def __init__(self, classes, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate, windowSize):
        self.net = Classifier_Net(classes, hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        # self.lossFunction = lossFunction
        self.lossFunction = nn.CrossEntropyLoss()
