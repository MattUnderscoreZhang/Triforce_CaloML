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
        self.input = nn.Linear(25 * 25 * 25, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.nHiddenLayers = nHiddenLayers
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons, 2)

    def forward(self, x):

	# ECAL input
	r = ECAL.view(-1, 25, 25, 25)
	model1 = nn.Conv3d(3, 4, 4, 4)(r)
	model1 = F.relu(model1)
	model1 = nn.MaxPool3D()(model1)
	model1 = Flatten()(model1)

	# HCAL input
	r = HCAL.view(-1, 5, 5, 60)
	model2 = nn.Conv3d(10, 2, 2, 6)(r)
	model2 = F.relu(model2)
	model2 = nn.MaxPool3D()(model2)
	model2 = Flatten()(model2)

	# join the two input models
	bmodel = merge([model1, model2], mode='concat')  # branched model

	# fully connected ending
	bmodel = (Dense(1000, activation='relu'))(bmodel)
	bmodel = (Dropout(0.5))(bmodel)

	return Dense(1, activation='linear', name='energy')(bmodel)  # output energy regression

class Regressor():
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate):
        self.net = Regressor_Net(hiddenLayerNeurons, nHiddenLayers, dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
