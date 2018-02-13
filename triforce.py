################################################################
#  ##########################################################  #
#  #                                                        #  #
#  #      /\\     TRIFORCE PARTICLE IDENTIFICATION SYSTEM   #  #
#  #     /__\\    Classification, Energy Regression & GAN   #  #
#  #    /\\ /\\                                             #  #
#  #   /__\/__\\                       Run using Python 3   #  #
#  #                                                        #  #
#  ##########################################################  #
################################################################

import torch
import torch.utils.data as data
from torch.autograd import Variable
import glob
import os
import numpy as np
import h5py as h5
import Loader.loader as loader
import sys

sys.dont_write_bytecode = True # prevent the creation of .pyc files

#########################
# Set tools and options #
#########################

from Options.default_options import *

####################################
# Load files and set up generators #
####################################

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

################
# Train models #
################

loss_history = []
avg_training_loss = 0.0
# test_loss = 0.0
epoch_test_loss = 0.0
endTraining = False
over_break_count = 0

for epoch in range(nEpochs):
    for i, data in enumerate(trainLoader):
        ECALs, HCALs, ys = data
        ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
        classifier.train(ECALs, HCALs, ys)
        regressor.train(ECALs, HCALs, ys)
        GAN.train(ECALs, HCALs, ys)
        # avg_training_loss += classifier.train(ECALs, ys)
        # if i % 20 == 19:
            # avg_training_loss /= 20 # average of loss over last 5 batches
            # print('[%d, %5d] train loss: %.10f' % (epoch+1, i+1, avg_training_loss)),
            # previous_test_loss = test_loss
            # test_loss = 0.0
            # # for data in testLoader:
            # nTestBatches = 10
            # for i in range(nTestBatches):
                # ECALs, HCALs, ys = data
                # ECALs, HCALs, ys = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda())
                # test_loss += classifier.eval(ECALs, ys)
            # test_loss = test_loss / nTestBatches
            # print(', test loss: %.10f' % (test_loss)),
            # loss_history.append([epoch + 1, i, avg_training_loss, test_loss])
            # avg_training_loss = 0.0
            # # decide whether or not to end training
            # relativeDeltaLoss = 1 if previous_test_loss==0 else (previous_test_loss - test_loss)/float(previous_test_loss)
            # print(', relative error: %.10f' % relativeDeltaLoss)
            # if (relativeDeltaLoss < relativeDeltaLossThreshold):
                # over_break_count+=1
            # else:
                # over_break_count=0
            # if (over_break_count >= relativeDeltaLossNumber):
                # endTraining = True
                # break
    previous_epoch_test_loss = epoch_test_loss
    classifier_test_loss = classifier.eval(ECALs, HCALs, ys)
    regressor_test_loss = regressor.eval(ECALs, HCALs, ys)
    GAN_test_loss = GAN.eval(ECALs, HCALs, ys)
    epoch_test_loss = classifier_test_loss + regressor_test_loss + GAN_test_loss
    relativeEpochDeltaLoss = 1 if previous_epoch_test_loss==0 else (previous_epoch_test_loss - epoch_test_loss)/float(previous_epoch_test_loss)
    if (relativeEpochDeltaLoss < relativeDeltaLossThreshold):
        endTraining = True
    if endTraining: break

if not os.path.exists(OutPath): os.makedirs(OutPath)
out_file = h5.File(OutPath+"Results.h5", 'w')
out_file.create_dataset("loss_history", data=np.array(loss_history))
torch.save(classifier.state_dict(), OutPath+"SavedModel")

print('Finished Training')

######################
# Analysis and plots #
######################

analyzer.analyze(classifier, testLoader, out_file)
