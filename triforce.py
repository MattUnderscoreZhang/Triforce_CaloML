##################################################
#   ▲   TRIFORCE PARTICLE IDENTIFICATION SYSTEM  #
#  ▲ ▲  classification, energy regression & GAN  #
##################################################

import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import loader
import glob
import os
import numpy as np
import h5py as h5

###############
# Set options #
###############

# basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/V3/Downsampled/EleChPi/"
# samplePath = [basePath + "ChPiEscan/ChPiEscan_*.h5", basePath + "EleEscan/EleEscan_*.h5"]
# classPdgID = [211, 11] # absolute IDs corresponding to paths above
basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/V3/Downsampled/GammaPi0/"
samplePath = [basePath + "Pi0Escan/Pi0Escan_*.h5", basePath + "GammaEscan/GammaEscan_*.h5"]
classPdgID = [111, 22] # absolute IDs corresponding to paths above
eventsPerFile = 10000

trainRatio = 0.66
nEpochs = 5 # break after this number of epochs
relativeDeltaLossThreshold = 0.001 # break if change in loss falls below this threshold over an entire epoch, or...
relativeDeltaLossNumber = 5 # ...for this number of test losses in a row
batchSize = 1000
nworkers = 0

OutPath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/SubmissionScripts/PyTorchNN/"+sys.argv[1]

learningRate = float(sys.argv[2])
decayRate = float(sys.argv[3])
dropoutProb = float(sys.argv[4])
hiddenLayerNeurons = int(sys.argv[5])
nHiddenLayers = int(sys.argv[6])

##############
# Load files #
##############

nParticles = len(samplePath)
particleFiles = [[]] * nParticles
for i, particlePath in enumerate(samplePath):
    particleFiles[i] = glob.glob(particlePath)

filesPerParticle = len(particleFiles[0])
nTrain = int(filesPerParticle * trainRatio)
nTest = filesPerParticle - nTrain
trainFiles = []
testFiles = []
for i in range(filesPerParticle):
    newFiles = []
    for j in range(nParticles):
        newFiles.append(particleFiles[j][i])
    if i < nTrain:
        trainFiles.append(newFiles)
    else:
        testFiles.append(newFiles)
eventsPerFile *= nParticles

trainSet = loader.HDF5Dataset(trainFiles, eventsPerFile, classPdgID)
testSet = loader.HDF5Dataset(testFiles, eventsPerFile, classPdgID)
trainLoader = data.DataLoader(dataset=trainSet,batch_size=batchSize,sampler=loader.OrderedRandomSampler(trainSet),num_workers=nworkers)
testLoader = data.DataLoader(dataset=testSet,batch_size=batchSize,sampler=loader.OrderedRandomSampler(testSet),num_workers=nworkers)

##################
# Classification #
##################

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(5 * 5 * 25 + 5 * 5 * 60, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons, 2)
    def forward(self, x1, x2):
        x1 = x1.view(-1, 5 * 5 * 25)
        x2 = x2.view(-1, 5 * 5 * 60)
        x = torch.cat([x1,x2], 1)
        x = self.input(x)
        for i in range(nHiddenLayers-1):
            x = F.relu(self.hidden(x))
            x = self.dropout(x)
        x = F.softmax(self.output(x))
        return x

classifier = Classification()
classifier.cuda()

# optimizer = optim.Adadelta(classifier.parameters(), lr=learningRate, weight_decay=decayRate)
optimizer = optim.Adam(classifier.parameters(), lr=learningRate, weight_decay=decayRate)
lossFunction = nn.CrossEntropyLoss()

###############
# Train model #
###############

loss_history = []
classifier.train() # set to training mode
avg_training_loss = 0.0
test_loss = 0.0
epoch_test_loss = 0.0
endTraining = False
over_break_count = 0
for epoch in range(nEpochs):
    for i, data in enumerate(trainLoader, 0):
        ECALs, HCALs, ys = data
        ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
        optimizer.zero_grad()
        outputs = classifier(ECALs, HCALs)
        loss = lossFunction(outputs, ys)
        loss.backward()
        optimizer.step()
        avg_training_loss += loss.data[0]
        if i % 20 == 19:
            avg_training_loss /= 20 # average of loss over last 5 batches
            print('[%d, %5d] train loss: %.10f' % (epoch+1, i+1, avg_training_loss)),
            previous_test_loss = test_loss
            test_loss = 0.0
            classifier.eval() # set to evaluation mode (turns off dropout)
            for data in testLoader:
                ECALs, HCALs, ys = data
                ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
                outputs = classifier(ECALs, HCALs)
                loss = lossFunction(outputs, ys)
                test_loss += loss.data[0]
            print(', test loss: %.10f' % (test_loss)),
            classifier.train() # set to training mode
            loss_history.append([epoch + 1, i, avg_training_loss, test_loss])
            avg_training_loss = 0.0
            # decide whether or not to end training
            relativeDeltaLoss = 1 if previous_test_loss==0 else (previous_test_loss - test_loss)/float(previous_test_loss)
            print(', relative error: %.10f' % relativeDeltaLoss)
            if (relativeDeltaLoss < relativeDeltaLossThreshold):
                over_break_count+=1
            else:
                over_break_count=0
            if (over_break_count >= relativeDeltaLossNumber):
                endTraining = True
                break
    previous_epoch_test_loss = epoch_test_loss
    epoch_test_loss = test_loss
    relativeEpochDeltaLoss = 1 if previous_epoch_test_loss==0 else (previous_epoch_test_loss - epoch_test_loss)/float(previous_epoch_test_loss)
    if (relativeEpochDeltaLoss < relativeDeltaLossThreshold):
        endTraining = True
    if endTraining: break

if not os.path.exists(OutPath): os.makedirs(OutPath)
file = h5.File(OutPath+"Results.h5", 'w')
file.create_dataset("loss_history", data=np.array(loss_history))

torch.save(classifier.state_dict(), OutPath+"SavedModel")

print('Finished Training')

######################
# Analysis and plots #
######################

correct = 0
total = 0
classifier.eval() # set to evaluation mode (turns off dropout)
for data in testLoader:
    ECALs, HCALs, ys = data
    ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
    outputs = classifier(ECALs, HCALs)
    _, predicted = torch.max(outputs.data, 1)
    total += ys.size(0)
    correct += (predicted == ys.data).sum()

file.create_dataset("outputs", data=np.array(outputs.data))

print('Accuracy of the network on test samples: %f %%' % (100 * float(correct) / total))
file.create_dataset("test_accuracy", data=np.array([100*float(correct)/total]))
