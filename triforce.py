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

from __future__ import division
import torch
import torch.utils.data as data
from torch.autograd import Variable
import glob, os, sys, shutil
import numpy as np
import h5py as h5
import Loader.loader as loader
from triforce_helper_functions import *
import Options
sys.dont_write_bytecode = True # prevent the creation of .pyc files

####################
# Set options file #
####################

optionsFileName = "default_options"

######################################################
# Import options & warn if options file has problems #
######################################################

# load options
exec("from Options." + optionsFileName + " import *")

# options file must have these parameters set
requiredOptionNames = ['samplePath', 'classPdgID', 'eventsPerFile', 'trainRatio', 'nEpochs', 'relativeDeltaLossThreshold', 'relativeDeltaLossNumber', 'batchSize', 'saveModelEveryNEpochs', 'outPath']
for optionName in requiredOptionNames:
    if optionName not in options.keys():
        print("ERROR: Please set", optionName, "in options file")
        sys.exit()

# if these parameters are not set, give them default values
defaultParameters = {'importGPU':False, 'nTrainMax':-1, 'nValidationMax':-1, 'nTestMax':-1, 'validationRatio':-1, 'nWorkers':0, 'calculate_test_loss_per_n_batches':20, 'test_loss_eval_max_n_batches':10}
for optionName in defaultParameters.keys():
    if optionName not in options.keys():
        options[optionName] = defaultParameters[optionName]

# for Caltech GPU cluster
if (options['importGPU']):
    import setGPU

# if validation parameters are not set, TriForce will use test set as validation set
if options['validationRatio'] == -1 and options['trainRatio'] >= 1:
    print("ERROR: trainRatio is too high")
elif options['validationRatio'] != -1 and options['trainRatio'] + options['validationRatio'] >= 1:
    print("ERROR: trainRatio and validationRatio are too high")

# warn if output directory already exists
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

# copy options file to output directory for logging purposes
optionsFilePath = Options.__file__[:Options.__file__.rfind('/')]
shutil.copyfile(optionsFilePath + "/" + optionsFileName + ".py", options['outPath']+"/options.h5")

####################################
# Load files and set up generators #
####################################

print('-------------------------------')
print('Loading Files')

# gather sample files for each type of particle
nClasses = len(options['samplePath'])
classFiles = [[]] * nClasses
for i, classPath in enumerate(options['samplePath']):
    classFiles[i] = glob.glob(classPath)

# calculate how many files of each particle type should be test, train, and validation
filesPerClass = len(classFiles[0])
nTrain = int(filesPerClass * options['trainRatio'])
nValidation = 0
if options['validationRatio']>0:
    nValidation = int(filesPerClass * options['validationRatio'])
nTest = filesPerClass - nTrain - nValidation
if options['nTrainMax']>0:
    nTrain = min(nTrain,options['nTrainMax'])
if options['nValidationMax']>0:
    nValidation = min(nValidation,options['nValidationMax'])
if options['nTestMax']>0:
    nTest = min(nTest,options['nTestMax'])
if (nTest==0 or nTrain==0 or (options['validationRatio']>0 and nValidation==0)):
    print("Not enough files found - check sample paths")

# list the train, test, and validation files
trainFiles = []
validationFiles = []
testFiles = []
for i in range(filesPerClass):
    newFiles = []
    for j in range(nClasses):
        newFiles.append(classFiles[j][i])
    if i < nTrain:
        trainFiles.append(newFiles)
    elif i < nTrain + nValidation:
        validationFiles.append(newFiles)
    else:
        testFiles.append(newFiles)
if options['validationRatio'] == -1:
    validationFiles = testFiles

# prepare the generators
trainSet = loader.HDF5Dataset(trainFiles, options['eventsPerFile']*nClasses, options['classPdgID'])
validationSet = loader.HDF5Dataset(validationFiles, options['eventsPerFile']*nClasses, options['classPdgID'])
testSet = loader.HDF5Dataset(testFiles, options['eventsPerFile']*nClasses, options['classPdgID'])
trainLoader = data.DataLoader(dataset=trainSet,batch_size=options['batchSize'],sampler=loader.OrderedRandomSampler(trainSet),num_workers=options['nWorkers'])
validationLoader = data.DataLoader(dataset=validationSet,batch_size=options['batchSize'],sampler=loader.OrderedRandomSampler(validationSet),num_workers=options['nWorkers'])
testLoader = data.DataLoader(dataset=testSet,batch_size=options['batchSize'],sampler=loader.OrderedRandomSampler(testSet),num_workers=options['nWorkers'])

############################################
# Data structures and options for training #
############################################

# loss and accuracy data e.g. history[LOSS][CLASSIFICATION][TRAIN][EPOCH]
history = [[[[[] for _ in range(2)] for _ in range(3)] for _ in range(3)] for _ in range(4)]

# enumerate parts of the data structure
qualifier_name = ['loss', 'accuracy', 'signalAccuracy', 'backgroundAccuracy']
LOSS, ACCURACY, SIGNAL_ACCURACY, BACKGROUND_ACCURACY = 0, 1, 2, 3
tools = [classifier, regressor, GAN]
tool_name = ['classifier', 'regressor', 'GAN']
tool_letter = ['C', 'R', 'G']
CLASSIFICATION, REGRESSION, _GAN = 0, 1, 2
split_name = ['train', 'validation', 'test']
TRAIN, VALIDATION, TEST = 0, 1, 2
period_name = ['batch', 'epoch']
BATCH, EPOCH = 0, 1

#######################
# Calculate test loss #
#######################

# determine when to end training
previous_total_test_loss = 0
previous_epoch_total_test_loss = 0
end_training = False
delta_loss_below_threshold_count = 0

def update_test_loss(epoch_end=False):
    
    # find the current test qualifiers e.g. test_qualifiers[LOSS][GAN]
    test_qualifiers = [[0]*3 for _ in range(4)]
    qualifier_index = [0, 1, 6, 7] # return indices for eval()
    for n_test_batches, data in enumerate(validationLoader):
        ECALs, HCALs, ys, energies = data
        ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
        for tool in range(len(tools)):
            if (tools[tool] != None):
                eval_results = eval(tools[tool], ECALs, HCALs, ys)
                for qualifier in range(len(qualifier_name)):
                    test_qualifiers[qualifier][tool] += eval_results[qualifier_index[qualifier]] 
        if (n_test_batches >= options['test_loss_eval_max_n_batches']):
            break
    for qualifier_index, test_qualifier in enumerate(test_qualifiers):
        test_qualifiers[qualifier_index] = [i/n_test_batches for i in test_qualifier]

    # print test loss and accuracy to screen
    print_prefix = ""
    if (epoch_end): print_prefix = "epoch "
    print(print_prefix + 'test loss:\t', end="")
    for tool in range(len(tools)):
        if (tools[tool] != None): print('(' + tool_letter[tool] + ') %.4f\t' % (test_qualifiers[LOSS][tool]), end="")
    print(print_prefix + 'test accuracy:\t', end="")
    for tool in range(len(tools)):
        if (tools[tool] != None): print('(' + tool_letter[tool] + ') %.4f\t' % (test_qualifiers[ACCURACY][tool]), end="")
    print()

    # save test qualifiers
    for tool in range(len(tools)):
        if (tools[tool] != None):
            for qualifier in range(len(qualifier_name)):
                history[qualifier][tool][TEST][BATCH].append(test_qualifiers[qualifier][tool])
                if (epoch_end): history[qualifier][tool][TEST][EPOCH].append(history[qualifier][tool])

    # decide whether or not to end training when this epoch finishes
    global previous_total_test_loss, previous_epoch_total_test_loss, delta_loss_below_threshold_count, end_training
    total_test_loss = 0
    for tool in range(len(tools)):
        if (tools[tool] != None): total_test_loss += test_qualifiers[LOSS][tool]
    relative_delta_loss = 1 if previous_total_test_loss==0 else (previous_total_test_loss - total_test_loss)/(previous_total_test_loss)
    previous_total_test_loss = total_test_loss
    delta_loss_below_threshold_count += (relative_delta_loss < options['relativeDeltaLossThreshold'])
    if (delta_loss_below_threshold_count >= options['relativeDeltaLossNumber']):
        end_training = True
    if (epoch_end):
        epoch_total_test_loss = test_qualifiers[LOSS][CLASSIFICATION] + test_qualifiers[LOSS][REGRESSION] + test_qualifiers[LOSS][_GAN]
        relative_delta_loss = 1 if previous_epoch_total_test_loss==0 else (previous_epoch_total_test_loss - epoch_total_test_loss)/(previous_epoch_total_test_loss)
        previous_epoch_total_test_loss = epoch_total_test_loss
        if (relative_delta_loss < options['relativeDeltaLossThreshold']):
            end_training = True

#########
# Train #
#########

print('Training')

for epoch in range(options['nEpochs']):
    epoch_loss = [0]*3
    epoch_accuracy = [0]*3
    for batch, data in enumerate(trainLoader):
        ECALs, HCALs, ys, energies = data
        ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
        if (tools[0] != None):
            epoch_loss[CLASSIFICATION] += train(tools[0], ECALs, HCALs, ys)[0]
            epoch_accuracy[CLASSIFICATION] += train(tools[0], ECALs, HCALs, ys)[1]
        if (tools[1] != None):
            epoch_loss[REGRESSION] += train(tools[1], ECALs, HCALs, energies)[0]
        if (tools[2] != None):
            epoch_loss[_GAN] += train(tools[2], ECALs, HCALs, ys)[0]
            epoch_accuracy[_GAN] += train(tools[2], ECALs, HCALs, ys)[1]
        if (batch+1) % options['calculate_test_loss_per_n_batches'] == 0:
            for tool in range(len(tools)):
                if (tools[tool] != None):
                    history[LOSS][tool][TRAIN][BATCH].append(epoch_loss[tool] / options['calculate_test_loss_per_n_batches'])
                    history[ACCURACY][tool][TRAIN][BATCH].append(epoch_accuracy[tool] / options['calculate_test_loss_per_n_batches'])
            print('-------------------------------')
            print('epoch %d, batch %d' % (epoch+1, batch+1))
            print('train loss:\t', end="")
            for tool in range(len(tools)):
                if (tools[tool] != None): print('(' + tool_letter[tool] + ') %.4f\t' % (history[LOSS][tool][TRAIN][BATCH][-1]), end="")
            print('train accuracy:\t', end="")
            for tool in range(len(tools)):
                if (tools[tool] != None): print('(' + tool_letter[tool] + ') %.4f\t' % (history[ACCURACY][tool][TRAIN][BATCH][-1]), end="")
            print()
            update_test_loss(epoch_end=False)
            epoch_loss = [0]*3
            epoch_accuracy = [0]*3
    if (tools[0] != None):
        history[ACCURACY][CLASSIFICATION][TRAIN][EPOCH].append(history[ACCURACY][CLASSIFICATION][TRAIN][BATCH][-1])
        history[LOSS][CLASSIFICATION][TRAIN][EPOCH].append(history[LOSS][CLASSIFICATION][TRAIN][BATCH][-1])
    update_test_loss(epoch_end=True)
    # save results
    if ((options['saveModelEveryNEpochs'] > 0) and ((epoch+1) % options['saveModelEveryNEpochs'] == 0)):
        if not os.path.exists(options['outPath']): os.makedirs(options['outPath'])
        for tool in range(len(tools)):
            if (tools[tool] != None): torch.save(tools[tool].net, options['outPath']+"saved_"+tool_name[tool]+"_epoch_"+str(epoch)+".pt")
    if end_training: break

print('-------------------------------')
print('Finished Training')

################
# Save results #
################

out_file = h5.File(options['outPath']+"results.h5", 'w')
for qualifier in [LOSS, ACCURACY, SIGNAL_ACCURACY, BACKGROUND_ACCURACY]:
    for tool in [CLASSIFICATION, REGRESSION, _GAN]:
        for split in [TRAIN, VALIDATION, TEST]:
            for period in [BATCH, EPOCH]:
                out_file.create_dataset(qualifier_name[qualifier]+"_"+tool_name[tool]+"_"+split_name[split]+"_"+period_name[period], data=np.array(history[qualifier][tool][split][period]))
for tool in range(len(tools)):
    if (tools[tool] != None): torch.save(tools[tool].net, options['outPath']+"saved_"+tool_name[tool]+".pt")

##########################
# Analyze and make plots #
##########################

print('Performing Analysis')
analyzer.analyze([tools[0], tools[1], tools[2]], validationLoader, out_file)
