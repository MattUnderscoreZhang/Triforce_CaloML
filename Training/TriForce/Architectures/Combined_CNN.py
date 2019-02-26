import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb # NOQA
from functools import reduce
from Architectures import LossFunctions


# helper functions for getting output sizes
def size_Conv3d(in_size, kernel_size, stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1]):
    out_size = 3 * [0]
    for i in range(3):
        out_size[i] = math.floor((in_size[i] + 2*padding[i] - dilation[i]*(kernel_size[i] - 1) - 1)/stride[i] + 1)
    return out_size


def size_MaxPool3d(in_size, kernel_size, stride=None, padding=[0, 0, 0], dilation=[1, 1, 1]):
    # default stride size is the kernel_size
    if stride is None:
        stride = kernel_size
    return size_Conv3d(in_size, kernel_size, stride, padding, dilation)


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
        self.hiddenLayerNeurons = options['hiddenLayerNeurons']
        self.nfiltECAL, self.kernelxyECAL, self.kernelzECAL = options['nfiltECAL'], options['kernelxyECAL'], options['kernelzECAL']
        self.nfiltHCAL, self.kernelxyHCAL, self.kernelzHCAL = options['nfiltHCAL'], options['kernelxyHCAL'], options['kernelzHCAL']
        self.maxpoolkernelECAL = options['maxpoolkernelECAL']
        self.maxpoolkernelHCAL = options['maxpoolkernelHCAL']
        self.inputScaleSumE = options['inputScaleSumE']
        self.inputScaleEta = options['inputScaleEta']
        self.inputScalePhi = options['inputScalePhi']

        self.outputs = []
        for particle_class in options['classPdgID']:
            self.outputs += [(str(particle_class)+"_classification", CLASSIFICATION)]
        self.outputs += [("energy_regression", REGRESSION), ("eta_regression", REGRESSION), ("phi_regression", REGRESSION)]

        # first layers: convolutions
        self.convECAL = nn.Conv3d(1, self.nfiltECAL, (self.kernelxyECAL, self.kernelxyECAL, self.kernelzECAL))
        self.maxpoolECAL = nn.MaxPool3d(self.maxpoolkernelECAL)
        self.convHCAL = nn.Conv3d(1, self.nfiltHCAL, (self.kernelxyHCAL, self.kernelxyHCAL, self.kernelzHCAL))
        self.maxpoolHCAL = nn.MaxPool3d(self.maxpoolkernelHCAL)

        # compute sizes for first dense layer
        sizes_ECAL = size_Conv3d([self.windowSizeECAL, self.windowSizeECAL, 25], [self.kernelxyECAL, self.kernelxyECAL, self.kernelzECAL])
        sizes_ECAL = size_MaxPool3d(sizes_ECAL, 3*[self.maxpoolkernelECAL])
        size_ECAL_flat = reduce(lambda x, y: x*y, sizes_ECAL) * self.nfiltECAL

        sizes_HCAL = size_Conv3d([self.windowSizeHCAL, self.windowSizeHCAL, 60], [self.kernelxyHCAL, self.kernelxyHCAL, self.kernelzHCAL])
        sizes_HCAL = size_MaxPool3d(sizes_HCAL, 3*[self.maxpoolkernelHCAL])
        size_HCAL_flat = reduce(lambda x, y: x*y, sizes_HCAL) * self.nfiltHCAL

        # dense layers
        self.hidden = nn.ModuleList()
        self.dropout = nn.ModuleList()
        for i in range(self.nHiddenLayers):
            if i == 0:
                # first layer after convolutions
                self.hidden.append(nn.Linear(size_ECAL_flat+size_HCAL_flat+4, self.hiddenLayerNeurons))
            else:
                self.hidden.append(nn.Linear(self.hiddenLayerNeurons, self.hiddenLayerNeurons))
            self.dropout.append(nn.Dropout(p=options['dropoutProb']))

        self.finalLayer = nn.Linear(self.hiddenLayerNeurons+5, len(self.outputs))

    def forward(self, data):

        # window slice and energy sums
        ECAL = Variable(data["ECAL"].cuda())
        lowerBound = math.ceil(ECAL.shape[1]/2) - int(math.ceil(self.windowSizeECAL/2))
        upperBound = lowerBound + self.windowSizeECAL
        ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
        ECAL = ECAL.contiguous().view(-1, 1, self.windowSizeECAL, self.windowSizeECAL, 25)
        ECAL_sum = torch.sum(ECAL.view(-1, self.windowSizeECAL * self.windowSizeECAL * 25), dim=1).view(-1, 1) * self.inputScaleSumE

        HCAL = Variable(data["HCAL"].cuda())
        lowerBound = math.ceil(HCAL.shape[1]/2) - int(math.ceil(self.windowSizeHCAL/2))
        upperBound = lowerBound + self.windowSizeHCAL
        HCAL = HCAL[:, lowerBound:upperBound, lowerBound:upperBound]
        HCAL = HCAL.contiguous().view(-1, 1, self.windowSizeHCAL, self.windowSizeHCAL, 60)
        HCAL_sum = torch.sum(HCAL.view(-1, self.windowSizeHCAL * self.windowSizeHCAL * 60), dim=1).view(-1, 1) * self.inputScaleSumE

        # get reco angles from event
        recoEta = Variable(data["recoEta"].cuda()).view(-1, 1) * self.inputScaleEta
        recoPhi = Variable(data["recoPhi"].cuda()).view(-1, 1) * self.inputScalePhi

        # ECAL convolutions
        branchECAL = self.convECAL(ECAL)
        branchECAL = F.relu(branchECAL)
        branchECAL = self.maxpoolECAL(branchECAL)
        # flatten
        branchECAL = branchECAL.view(branchECAL.size(0), -1)

        # HCAL convolutions
        branchHCAL = self.convHCAL(HCAL)
        branchHCAL = F.relu(branchHCAL)
        branchHCAL = self.maxpoolHCAL(branchHCAL)
        # flatten
        branchHCAL = branchHCAL.view(branchHCAL.size(0), -1)

        # join the two branches and angles, energy sums
        x = torch.cat((branchECAL, branchHCAL, recoPhi, recoEta, ECAL_sum, HCAL_sum), 1)
        # fully connected layers
        for i in range(self.nHiddenLayers):
            x = F.relu(self.hidden[i](x))
            x = self.dropout[i](x)
        # cat angles / energy sums back in before final layer
        x = torch.cat([x, recoPhi, recoEta, ECAL_sum, HCAL_sum, torch.ones([data['ECAL'].shape[0], 1]).cuda()], 1)
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


class Net():
    def __init__(self, options):
        self.net = Classifier_Net(options)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=options['learningRate'], weight_decay=options['decayRate'])
        self.lossFunction = LossFunctions.combinedLossFunction
