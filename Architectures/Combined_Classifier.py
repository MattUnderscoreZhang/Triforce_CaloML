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
        # settings
        self.windowSize = windowSize
        self.nHiddenLayers = nHiddenLayers
        self.outputs = []
        for particle_class in classes:
            self.outputs += [(str(particle_class)+"_classification", CLASSIFICATION)]
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION)]
        # layers
        self.input = nn.Linear(windowSize * windowSize * 25, hiddenLayerNeurons)
        self.hidden = [None] * self.nHiddenLayers
        self.dropout = [None] * self.nHiddenLayers
        for i in range(self.nHiddenLayers):
            self.hidden[i] = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
            self.hidden[i].cuda()
            self.dropout[i] = nn.Dropout(p = dropoutProb)
            self.dropout[i].cuda()
        self.finalLayer = nn.Linear(hiddenLayerNeurons, len(self.outputs)) # nClasses = 2 for binary classifier

    def forward(self, data):
        # window slice
        x = Variable(data["ECAL"].cuda())
        lowerBound = 26 - int(math.ceil(self.windowSize/2))
        upperBound = lowerBound + self.windowSize
        x = x[:, lowerBound:upperBound, lowerBound:upperBound]
        x = x.contiguous().view(-1, self.windowSize * self.windowSize * 25)
        # feed forward
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden[i](x))
            x = self.dropout[i](x)
        x = self.finalLayer(x)
        # preparing output
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

def lossFunction(output, data, term_weights):
    # classification loss: cross entropy
    loss_class = term_weights['classification'] * F.cross_entropy(output['classification'], Variable(data['pdgID'].cuda()))
    # regression loss: mse
    # to add: per-event weights for energy regression
    loss_energy = term_weights['energy_regression'] * F.mse_loss(output['energy_regression'], Variable(data['energy'].cuda()))
    loss_eta = term_weights['eta_regression'] * F.mse_loss(output['eta_regression'], Variable(data['eta'].cuda()))
    return loss_class+loss_energy+loss_eta

class Net():
    def __init__(self, classes, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate, windowSize):
        self.net = Classifier_Net(classes, hiddenLayerNeurons, nHiddenLayers, dropoutProb, windowSize)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = lossFunction
