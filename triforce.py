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
import glob, os, sys
import numpy as np
import h5py as h5
import Loader.loader as loader
import shutil
from triforce_helper_functions import *
import Options

sys.dont_write_bytecode = True # prevent the creation of .pyc files

#####################
# Load options file #
#####################

optionsFileName = "default_options"

#########################
# Set tools and options #
#########################

exec("from Options." + optionsFileName + " import *")

optionNames = ['samplePath', 'classPdgID', 'eventsPerFile', 'nWorkers', 'trainRatio', 'nEpochs', 'relativeDeltaLossThreshold', 'relativeDeltaLossNumber', 'batchSize', 'saveModelEveryNEpochs', 'outPath', 'nTrainMax',"nTestMax"]

for optionName in optionNames:
    if optionName not in options.keys():
        print("ERROR: Please set", optionName, "in options file")
        sys.exit()

if not os.path.exists(options['outPath']):
    os.makedirs(options['outPath'])
else:
    print("WARNING: Output directory already exists - overwrite (y/n)?")
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    while True:
        choice = input().lower()
        if choice in valid:
            overwrite = valid[choice]
            break
        else:
            print("Please respond with 'yes' or 'no'")
    if overwrite:
        shutil.rmtree(options['outPath'])
        os.makedirs(options['outPath'])
    else:
        sys.exit()

optionsFilePath = Options.__file__[:Options.__file__.rfind('/')]
shutil.copyfile(optionsFilePath + "/" + optionsFileName + ".py", options['outPath']+"/options.h5")

####################################
# Load files and set up generators #
####################################

print('-------------------------------')
print('Loading Files')

nParticles = len(options['samplePath'])
particleFiles = [[]] * nParticles
for i, particlePath in enumerate(options['samplePath']):
    particleFiles[i] = glob.glob(particlePath)

filesPerParticle = len(particleFiles[0])
nTrain = int(filesPerParticle * options['trainRatio'])
nTest = filesPerParticle - nTrain
if (nTest==0 or nTrain==0):
    print("Not enough files found - check sample paths")
nTrain = max(nTrain,options['nTrainMax'])
nTest = max(nTest,options['nTestMax'])
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
options['eventsPerFile'] *= nParticles

trainSet = loader.HDF5Dataset(trainFiles, options['eventsPerFile'], options['classPdgID'])
testSet = loader.HDF5Dataset(testFiles, options['eventsPerFile'], options['classPdgID'])
trainLoader = data.DataLoader(dataset=trainSet,batch_size=options['batchSize'],sampler=loader.OrderedRandomSampler(trainSet),num_workers=options['nWorkers'])
testLoader = data.DataLoader(dataset=testSet,batch_size=options['batchSize'],sampler=loader.OrderedRandomSampler(testSet),num_workers=options['nWorkers'])

################
# Train models #
################
classifier_accuracy_epoch_train = []
classifier_loss_epoch_train = []
classifier_loss_history_train = []
regressor_loss_history_train = []
GAN_loss_history_train = []
classifier_accuracy_history_train = []
GAN_accuracy_history_train = []
classifier_loss_history_test = []
regressor_loss_history_test = []
GAN_loss_history_test = []
classifier_accuracy_history_test = []
GAN_accuracy_history_test = []

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
    classifier_test_accuracy = 0
    GAN_test_accuracy = 0
    n_test_batches = 0
    for data in testLoader:
        ECALs, HCALs, ys, energies = data
        ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
        if (classifier != None):
            classifier_test_loss += eval(classifier, ECALs, HCALs, ys)[0]
            classifier_test_accuracy += eval(classifier, ECALs, HCALs, ys)[1]
        if (regressor != None):
            regressor_test_loss += eval(regressor, ECALs, HCALs, energies)[0]
        if (GAN != None):
            GAN_test_loss += eval(GAN, ECALs, HCALs, ys)[0]
            GAN_test_accuracy += eval(GAN, ECALs, HCALs, ys)[1]
        n_test_batches += 1
        if (n_test_batches >= max_n_test_batches):
            break
    classifier_test_loss /= n_test_batches
    regressor_test_loss /= n_test_batches
    GAN_test_loss /= n_test_batches
    classifier_test_accuracy /= n_test_batches
    GAN_test_accuracy /= n_test_batches
    if (not epoch_end):
        print('test loss:\t', end="")
        if (classifier != None): print('(C) %.4f\t' % (classifier_test_loss), end="")
        if (regressor != None): print('(R) %.4f\t' % (regressor_test_loss), end="")
        if (GAN != None): print('(G) %.4f\t' % (GAN_test_loss), end="")
        print()
        print('test accuracy:\t', end="")
        if (classifier != None): print('(C) %.4f\t' % (classifier_test_accuracy), end="")
        if (regressor != None): print('(R) -----\t', end="")
        if (GAN != None): print('(G) %.4f\t' % (GAN_test_accuracy), end="")
        print()
    else:
        print('epoch test loss:\t', end="")
        if (classifier != None): print('(C) %.4f\t' % (classifier_test_loss), end="")
        if (regressor != None): print('(R) %.4f\t' % (regressor_test_loss), end="")
        if (GAN != None): print('(G) %.4f\t' % (GAN_test_loss), end="")
        print()
        print('epoch test accuracy:\t', end="")
        if (classifier != None): print('(C) %.4f\t' % (classifier_test_accuracy), end="")
        if (regressor != None): print('(R) -----\t', end="")
        if (GAN != None): print('(G) %.4f\t' % (GAN_test_accuracy), end="")
        print()
    classifier_loss_history_train.append(classifier_test_loss)
    regressor_loss_history_train.append(regressor_test_loss)
    GAN_loss_history_train.append(GAN_test_loss)
    classifier_accuracy_history_train.append(classifier_test_accuracy)
    GAN_accuracy_history_train.append(GAN_test_accuracy)
    # decide whether or not to end training when this epoch finishes
    if (not epoch_end):
        total_test_loss = 0
        if (classifier != None): total_test_loss += classifier_test_loss
        if (regressor != None): total_test_loss += regressor_test_loss
        if (GAN != None): total_test_loss += GAN_test_loss
        relativeDeltaLoss = 1 if previous_total_test_loss==0 else (previous_total_test_loss - total_test_loss)/(previous_total_test_loss)
        previous_total_test_loss = total_test_loss
        if (relativeDeltaLoss < options['relativeDeltaLossThreshold']):
            over_break_count += 1
        else:
            over_break_count = 0
        if (over_break_count >= options['relativeDeltaLossNumber']):
            end_training = True
    else:
        epoch_total_test_loss = classifier_test_loss + regressor_test_loss + GAN_test_loss
        relativeDeltaLoss = 1 if previous_epoch_total_test_loss==0 else (previous_epoch_total_test_loss - epoch_total_test_loss)/(previous_epoch_total_test_loss)
        previous_epoch_total_test_loss = epoch_total_test_loss
        if (relativeDeltaLoss < options['relativeDeltaLossThreshold']):
            end_training = True

# perform training
print('Training')
for epoch in range(options['nEpochs']):
    reserve_for_accuracy_epoch_train = []
    reserve_for_loss_epoch_train = []
    classifier_training_loss = 0
    regressor_training_loss = 0
    GAN_training_loss = 0
    classifier_training_accuracy = 0
    GAN_training_accuracy = 0
    for i, data in enumerate(trainLoader):
        ECALs, HCALs, ys, energies = data
        ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
        if (classifier != None):
            classifier_training_loss += train(classifier, ECALs, HCALs, ys)[0]
            classifier_training_accuracy += train(classifier, ECALs, HCALs, ys)[1]
        if (regressor != None):
            regressor_training_loss += train(regressor, ECALs, HCALs, energies)[0]
        if (GAN != None):
            GAN_training_loss += train(GAN, ECALs, HCALs, ys)[0]
            GAN_training_accuracy += train(GAN, ECALs, HCALs, ys)[1]
        if i % calculate_loss_per == calculate_loss_per - 1:
            reserve_for_accuracy_epoch_train.append(classifier_training_accuracy / calculate_loss_per)
            reserve_for_loss_epoch_train.append(classifier_training_loss / calculate_loss_per)
            classifier_loss_history_train.append(classifier_training_loss / calculate_loss_per)
            regressor_loss_history_train.append(regressor_training_loss / calculate_loss_per)
            GAN_loss_history_train.append(GAN_training_loss / calculate_loss_per)
            classifier_accuracy_history_train.append(classifier_training_accuracy / calculate_loss_per)
            GAN_accuracy_history_train.append(GAN_training_accuracy / calculate_loss_per)
            print('-------------------------------')
            print('epoch %d, batch %d' % (epoch+1, i+1))
            print('train loss:\t', end="")
            if (classifier != None): print('(C) %.4f\t' % (classifier_loss_history_train[-1]), end="")
            if (regressor != None): print('(R) %.4f\t' % (regressor_loss_history_train[-1]), end="")
            if (GAN != None): print('(G) %.4f\t' % (GAN_loss_history_train[-1]), end="")
            print()
            print('train accuracy:\t', end="")
            if (classifier != None): print('(C) %.4f\t' % (classifier_accuracy_history_train[-1]), end="")
            if (regressor != None): print('(R) -----\t', end="")
            if (GAN != None): print('(G) %.4f\t' % (GAN_accuracy_history_train[-1]), end="")
            print()
            update_test_loss(epoch_end=False)
            classifier_training_loss = 0
            regressor_training_loss = 0
            GAN_training_loss = 0
            classifier_training_accuracy = 0
            GAN_training_accuracy = 0
    classifier_accuracy_epoch_train.append(float(sum(reserve_for_accuracy_epoch_train)) / max(len(reserve_for_accuracy_epoch_train), 1))
    classifier_loss_epoch_train.append(float(sum(reserve_for_loss_epoch_train)) / max(len(reserve_for_loss_epoch_train), 1))
    update_test_loss(epoch_end=True)
    # save results
    if ((options['saveModelEveryNEpochs'] > 0) and ((epoch+1) % options['saveModelEveryNEpochs'] == 0)):
        if not os.path.exists(options['outPath']): os.makedirs(options['outPath'])
        if (classifier != None): 
            torch.save(classifier.net, options['outPath']+"saved_classifier_epoch_"+str(epoch)+".pt")
        if (regressor != None): 
            torch.save(regressor.net, options['outPath']+"saved_regressor_epoch_"+str(epoch)+".pt")
        if (GAN != None): 
            torch.save(GAN.net, options['outPath']+"saved_GAN_epoch_"+str(epoch)+".pt")
    if end_training: break

# save results
out_file = h5.File(options['outPath']+"results.h5", 'w')
if (classifier != None): 
    out_file.create_dataset("classifier_accuracy_epoch_train", data=np.array(classifier_accuracy_epoch_train))
    out_file.create_dataset("classifier_accuracy_epoch_train", data=np.array(classifier_loss_epoch_train))
    out_file.create_dataset("classifier_accuracy_history_train", data=np.array(classifier_accuracy_history_train))
    out_file.create_dataset("classifier_loss_history_train", data=np.array(classifier_loss_history_train))
    out_file.create_dataset("classifier_loss_history_test", data=np.array(classifier_loss_history_test))
    torch.save(classifier.net, options['outPath']+"saved_classifier.pt")
if (regressor != None): 
    out_file.create_dataset("regressor_loss_history_train", data=np.array(regressor_loss_history_train))
    out_file.create_dataset("regressor_loss_history_test", data=np.array(regressor_loss_history_test))
    torch.save(regressor.net, options['outPath']+"saved_regressor.pt")
if (GAN != None): 
    out_file.create_dataset("GAN_loss_history_train", data=np.array(GAN_loss_history_train))
    out_file.create_dataset("GAN_loss_history_test", data=np.array(GAN_loss_history_test))
    torch.save(GAN.net, options['outPath']+"saved_GAN.pt")

print('-------------------------------')
print('Finished Training')

######################
# Analysis and plots #
######################

print('Performing Analysis')
analyzer.analyze([classifier, regressor, GAN], testLoader, out_file)

import Tools.Classification_Plotter as Classification_Plotter

Classification_Plotter.make_all(options['outPath']+'results.h5')
