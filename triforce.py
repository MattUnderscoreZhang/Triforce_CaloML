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
from triforce_helper_functions import *

sys.dont_write_bytecode = True # prevent the creation of .pyc files

#########################
# Set tools and options #
#########################

from Options.default_options import *

####################################
# Load files and set up generators #
####################################

print('Loading Files')

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

classifier_loss_history_train = []
regressor_loss_history_train = []
GAN_loss_history_train = []
classifier_loss_history_test = []
regressor_loss_history_test = []
GAN_loss_history_test = []

calculate_loss_per = 20
max_n_test_batches = 10 # stop evaluating test loss after this many batches
previous_total_test_loss = 0 # stop training when loss stops decreasing
previous_epoch_total_test_loss = 0

end_training = False
over_break_count = 0

# see what the current test loss is, and whether we should keep training
def update_test_loss(epoch_end):
    global previous_total_test_loss, previous_epoch_total_test_loss, over_break_count, end_training
    classifier_test_loss = 0
    regressor_test_loss = 0
    GAN_test_loss = 0
    n_test_batches = 0
    for data in testLoader:
        ECALs, HCALs, ys, energies = data
        ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
        if (classifier != None): classifier_test_loss += eval(classifier, ECALs, HCALs, ys)[0]
        if (regressor != None): regressor_test_loss += eval(regressor, ECALs, HCALs, energies)[0]
        if (GAN != None): GAN_test_loss += eval(GAN, ECALs, HCALs, ys)[0]
        n_test_batches += 1
        if (n_test_batches >= max_n_test_batches):
            break
    classifier_test_loss /= n_test_batches
    regressor_test_loss /= n_test_batches
    GAN_test_loss /= n_test_batches
    if (not epoch_end):
        print(' - test loss - (C) %.4f, (R) %.4f, (G) %.4f' % (classifier_test_loss, regressor_test_loss, GAN_test_loss))
    else:
        print('epoch test loss - (C) %.4f, (R) %.4f, (G) %.4f' % (classifier_test_loss, regressor_test_loss, GAN_test_loss))
    classifier_loss_history_train.append(classifier_test_loss)
    regressor_loss_history_train.append(regressor_test_loss)
    GAN_loss_history_train.append(GAN_test_loss)
    # decide whether or not to end training when this epoch finishes
    if (not epoch_end):
        total_test_loss = 0
        if (classifier != None): total_test_loss += classifier_test_loss
        if (regressor != None): total_test_loss += regressor_test_loss
        if (GAN != None): total_test_loss += GAN_test_loss
        relativeDeltaLoss = 1 if previous_total_test_loss==0 else (previous_total_test_loss - total_test_loss)/(previous_total_test_loss)
        previous_total_test_loss = total_test_loss
        if (relativeDeltaLoss < relativeDeltaLossThreshold):
            over_break_count += 1
        else:
            over_break_count = 0
        if (over_break_count >= relativeDeltaLossNumber):
            end_training = True
    else:
        epoch_total_test_loss = classifier_test_loss + regressor_test_loss + GAN_test_loss
        relativeDeltaLoss = 1 if previous_epoch_total_test_loss==0 else (previous_epoch_total_test_loss - epoch_total_test_loss)/(previous_epoch_total_test_loss)
        previous_epoch_total_test_loss = epoch_total_test_loss
        if (relativeDeltaLoss < relativeDeltaLossThreshold):
            end_training = True

# perform training
print('Training')
for epoch in range(nEpochs):
    classifier_training_loss = 0
    regressor_training_loss = 0
    GAN_training_loss = 0
    for i, data in enumerate(trainLoader):
        ECALs, HCALs, ys, energies = data
        ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
        if (classifier != None): classifier_training_loss += train(classifier, ECALs, HCALs, ys)[0]
        if (regressor != None): regressor_training_loss += train(regressor, ECALs, HCALs, energies)[0]
        if (GAN != None): GAN_training_loss += train(GAN, ECALs, HCALs, ys)[0]
        if i % calculate_loss_per == calculate_loss_per - 1:
            classifier_loss_history_train.append(classifier_training_loss / calculate_loss_per)
            regressor_loss_history_train.append(regressor_training_loss / calculate_loss_per)
            GAN_loss_history_train.append(GAN_training_loss / calculate_loss_per)
            print('epoch %d, batch %d - train loss - (C) %.4f, (R) %.4f, (G) %.4f' % (epoch+1, i+1, classifier_loss_history_train[-1], regressor_loss_history_train[-1], GAN_loss_history_train[-1]), end="")
            update_test_loss(epoch_end=False)
            classifier_training_loss = 0
            regressor_training_loss = 0
            GAN_training_loss = 0
    update_test_loss(epoch_end=True)
    if end_training: break

# save results
if not os.path.exists(OutPath): os.makedirs(OutPath)
out_file = h5.File(OutPath+"results.h5", 'w')
if (classifier != None): 
    out_file.create_dataset("classifier_loss_history_train", data=np.array(classifier_loss_history_train))
    out_file.create_dataset("classifier_loss_history_test", data=np.array(classifier_loss_history_test))
    torch.save(classifier.net, OutPath+"saved_classifier.pt")
if (regressor != None): 
    out_file.create_dataset("regressor_loss_history_train", data=np.array(regressor_loss_history_train))
    out_file.create_dataset("regressor_loss_history_test", data=np.array(regressor_loss_history_test))
    torch.save(regressor.net, OutPath+"saved_regressor.pt")
if (GAN != None): 
    out_file.create_dataset("GAN_loss_history_train", data=np.array(GAN_loss_history_train))
    out_file.create_dataset("GAN_loss_history_test", data=np.array(GAN_loss_history_test))
    torch.save(GAN.net, OutPath+"saved_GAN.pt")

print('Finished Training')

######################
# Analysis and plots #
######################

print('Performing Analysis')
analyzer.analyze([classifier, regressor, GAN], testLoader, out_file)
print('Finished - Have a Nice Day!')
