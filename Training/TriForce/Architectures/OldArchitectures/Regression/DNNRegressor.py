import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############
# Regression #
##############

class Regressor_Net(nn.Module):
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb):
        super().__init__()
        self.input = nn.Linear(51 * 51 * 25 + 11 * 11 * 60, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.nHiddenLayers = nHiddenLayers
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons, 1)
    def forward(self, x1, x2):
        x1 = x1.view(-1, 51 * 51 * 25)
        x2 = x2.view(-1, 11 * 11 * 60)
        x = torch.cat([x1, x2], 1)
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden(x))
            x = self.dropout(x)
        x = self.output(x)
        return x
class Regressor():
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate):
        self.net = Regressor_Net(hiddenLayerNeurons, nHiddenLayers, dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.MSELoss()
