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

    def __init__(self, options):
        super().__init__()
        # settings
        self.windowSizeECAL = options['windowSizeECAL']
        self.windowSizeHCAL = options['windowSizeHCAL']
        self.nHiddenLayers = options['nHiddenLayers']
        hiddenLayerNeurons = options['hiddenLayerNeurons']
        self.outputs = []
        for particle_class in options['classPdgID']:
            self.outputs += [(str(particle_class)+"_classification", CLASSIFICATION)]
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION)]
        # layers
        self.input = nn.Linear(self.windowSizeECAL * self.windowSizeECAL * 25 + self.windowSizeHCAL * self.windowSizeHCAL * 60 + 2, hiddenLayerNeurons)
        self.hidden = [None] * self.nHiddenLayers
        self.dropout = [None] * self.nHiddenLayers
        for i in range(self.nHiddenLayers):
            self.hidden[i] = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
            self.hidden[i].cuda()
            self.dropout[i] = nn.Dropout(p = options['dropoutProb'])
            self.dropout[i].cuda()
        self.finalLayer = nn.Linear(hiddenLayerNeurons + 2, len(self.outputs)) # nClasses = 2 for binary classifier
        # initialize weights for energy sums in energy output to 1: assume close to identity
        energy_index = self.outputs.index(("energy_regression", REGRESSION))
        output_params = self.finalLayer.weight.data
        output_params[energy_index][-1] = 1.0
        output_params[energy_index][-2] = 1.0

    def forward(self, data):
        # window slice and energy sums
        ECAL = Variable(data["ECAL"].cuda())
        lowerBound = 26 - int(math.ceil(self.windowSizeECAL/2))
        upperBound = lowerBound + self.windowSizeECAL
        ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
        ECAL = ECAL.contiguous().view(-1, self.windowSizeECAL * self.windowSizeECAL * 25)
        ECAL_sum = torch.sum(ECAL, dim = 1).view(-1, 1)

        HCAL = Variable(data["HCAL"].cuda())
        lowerBound = 6 - int(math.ceil(self.windowSizeHCAL/2))
        upperBound = lowerBound + self.windowSizeHCAL
        HCAL = HCAL[:, lowerBound:upperBound, lowerBound:upperBound]
        HCAL = HCAL.contiguous().view(-1, self.windowSizeHCAL * self.windowSizeHCAL * 60)
        HCAL_sum = torch.sum(HCAL, dim = 1).view(-1, 1)

        x = torch.cat([ECAL, HCAL, ECAL_sum, HCAL_sum], 1)
        # feed forward
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden[i](x))
            x = self.dropout[i](x)
        # cat energy sums back in before final layer
        x = torch.cat([x, ECAL_sum, HCAL_sum], 1)
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

def weighted_mse_loss(pred,target,weights):
    sqerr = (pred-target)**2
    sqerr = sqerr * weights
    loss = torch.mean(sqerr, dim=0)
    return loss

def lossFunction(output, data, term_weights):
    # classification loss: cross entropy
    loss_class = term_weights['classification'] * F.cross_entropy(output['classification'], Variable(data['pdgID'].cuda()))
    # regression loss: mse
    truth_energy = Variable(data['energy'].cuda())
    # use per-event weights for energy to emphasize lower energies
    event_weights = 1.0 / torch.log(truth_energy)
    loss_energy = term_weights['energy_regression'] * weighted_mse_loss(output['energy_regression'], truth_energy, event_weights)
    loss_eta = term_weights['eta_regression'] * F.mse_loss(output['eta_regression'], Variable(data['eta'].cuda()))
    return loss_class+loss_energy+loss_eta

class Net():
    def __init__(self, options):
        self.net = Classifier_Net(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = lossFunction
