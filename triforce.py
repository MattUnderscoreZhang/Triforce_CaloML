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
import glob, os, sys, shutil, socket
sys.dont_write_bytecode = True # prevent the creation of .pyc files
import numpy as np
import h5py as h5
import Loader.loader as loader
from Loader import transforms
import Options
import pdb
from timeit import default_timer as timer
from Trainers import ClassRegTrainer

# os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7, 8, 9'

start = timer()

# for running at caltech
if 'culture-plate' in socket.gethostname():
    import setGPU

####################
# Set options file #
####################

optionsFileName = "combined"

######################################################
# Import options & warn if options file has problems #
######################################################

# load options
exec("from Options." + optionsFileName + " import *")

# options file must have these parameters set
requiredOptionNames = ['samplePath', 'classPdgID', 'trainRatio', 'nEpochs', 'microBatchSize', 'outPath']
for optionName in requiredOptionNames:
    if optionName not in options.keys():
        print("ERROR: Please set", optionName, "in options file")
        sys.exit()

# if these parameters are not set, give them default values
defaultParameters = {'importGPU':False, 'nTrainMax':-1, 'nValidationMax':-1, 'nTestMax':-1, 'validationRatio':0, 'nMicroBatchesInMiniBatch':1, 'nWorkers':0, 'test_loss_eval_max_n_batches':10, 'earlyStopping':False, 'relativeDeltaLossThreshold':0, 'relativeDeltaLossNumber':5, 'saveModelEveryNEpochs':0, 'saveFinalModel':0}
for optionName in defaultParameters.keys():
    if optionName not in options.keys():
        options[optionName] = defaultParameters[optionName]

# if validation parameters are not set, TriForce will use test set as validation set
if options['validationRatio'] + options['trainRatio'] >= 1:
    print("ERROR: validationRatio and/or trainRatio is too high")

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

# copy code to output directory for logging purposes
optionsFilePath = Options.__file__[:Options.__file__.rfind('/')]
shutil.copyfile(optionsFilePath + "/" + optionsFileName + ".py", options['outPath']+"/options.py")
shutil.copyfile(optionsFilePath + "/../triforce.py", options['outPath']+"/triforce.py")
# also write options dict to store also command line input parameters
with open(options['outPath']+"/options_dict.py", 'w') as f:
    f.write('options = '+str(options))

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

# determine numbers of test, train, and validation files
filesPerClass = min([len(files) for files in classFiles])
nTrain = int(filesPerClass * options['trainRatio'])
nValidation = int(filesPerClass * options['validationRatio'])
nTest = filesPerClass - nTrain - nValidation
if options['nTrainMax']>0:
    nTrain = min(nTrain,options['nTrainMax'])
if options['nValidationMax']>0:
    nValidation = min(nValidation,options['nValidationMax'])
if options['nTestMax']>0:
    nTest = min(nTest,options['nTestMax'])
if (options['validationRatio'] > 0):
    print("Split (files per class): %d train, %d test, %d validation" % (nTrain, nTest, nValidation))
else:
    print("Split (files per class): %d train, %d test" % (nTrain, nTest))
if (nTest==0 or nTrain==0 or (options['validationRatio']>0 and nValidation==0)):
    print("Not enough files found - check sample paths")
    sys.exit()
print('-------------------------------')

# split the train, test, and validation files
# get lists of [[class1_file1, class2_file1], [class1_file2, class2_file2], ...]
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
    elif i < nTrain + nValidation + nTest:
        testFiles.append(newFiles)
    else:
        break
if options['validationRatio'] == 0:
    validationFiles = testFiles

# prepare the generators
print('Defining training dataset')
trainSet = loader.HDF5Dataset(trainFiles, options['classPdgID'], options['filters'])
print('Defining validation dataset')
validationSet = loader.HDF5Dataset(validationFiles, options['classPdgID'], options['filters'])
print('Defining test dataset')
testSet = loader.HDF5Dataset(testFiles, options['classPdgID'], options['filters'])
trainLoader = data.DataLoader(dataset=trainSet,batch_size=options['microBatchSize'],sampler=loader.OrderedRandomSampler(trainSet),num_workers=options['nWorkers'])
validationLoader = data.DataLoader(dataset=validationSet,batch_size=options['microBatchSize'],sampler=loader.OrderedRandomSampler(validationSet),num_workers=options['nWorkers'])
testLoader = data.DataLoader(dataset=testSet,batch_size=options['microBatchSize'],sampler=loader.OrderedRandomSampler(testSet),num_workers=options['nWorkers'])
print('-------------------------------')

################################################
# Data structures for holding training results #
################################################

class historyData(list):
    def __init__(self, my_list=[]):
        super().__init__(my_list)
    def extend(self, places):
        for _ in range(places):
            self.append(historyData())
    def __getitem__(self, key): # overloads list[i] for i out of range
        if key >= len(self):
            self.extend(key+1-len(self))
        return super().__getitem__(key)
    def __setitem__(self, key, item): # overloads list[i]=x for i out of range
        if key >= len(self):
            self.extend(key+1-len(self))
        super().__setitem__(key, item)
    def __iadd__(self, key): # overloads []+=x
        return key
    def __add__(self, key): # overloads []+x
        return key

# training results e.g. history[CLASS_LOSS][TRAIN][EPOCH]
GAN_history = historyData()

# enumerate parts of the data structure
CLASS_LOSS, CLASS_ACC = 0, 1
GAN_stat_name = ['discriminator_loss', 'discriminator_acc', 'generator_loss', 'discriminator_acc_on_generator_data']
DISCRIM_LOSS, DISCRIM_ACC, GEN_LOSS, GEN_ACC = 0, 1, 2, 3
split_name = ['train', 'validation', 'test']
TRAIN, VALIDATION, TEST = 0, 1, 2
timescale_name = ['batch', 'epoch']
BATCH, EPOCH = 0, 1

###############################
# Load or Train Class+Reg Net #
###############################

if options['skipClassRegTrain']:
    print('Loading Classifier and Regressor')
    # check if there is a state_dict of a trained model. 
    if os.path.exists(options['outPath']+"saved_classifier.pt"):
        combined_classifier.net.load_state_dict(torch.load(options['outPath']+"saved_classifier.pt")) 
    else: 
        print('WARNING: Found no trained models. Train new model (y/n)?')
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        while True:
            choice = input().lower()
            if choice in valid:
                overwrite = valid[choice]
                break
            else:
                print("Please respond with 'yes' or 'no'")
        if overwrite:
            print('-------------------------------')
            print('Training Classifier and Regressor')
            options['skipClassRegTrain'] = False
        else:
            sys.exit()

if options['skipClassRegTrain']: 
    print('-------------------------------')
    print('Training Classifier and Regressor')
    class_reg_trainer = ClassRegTrainer.ClassRegTrainer(combined_classifier)
    class_reg_trainer.train() 
print('-------------------------------')

################
# GAN Training #
################

# def GAN_training():

    # for epoch in range(options['nEpochs']):

        # for data_train in trainLoader:
            # discriminator.net.train()
            # outputs = discriminator.net(data_train)
            # disc_loss = discriminator.lossFunction(outputs, torch.Tensor([1]))
            # discriminator.optimizer.zero_grad()
            # disc_loss.backward()
            # discriminator.optimizer.step()

# print('Training GAN')
# GAN_training()
# print('-------------------------------')
# print('Finished Training')

################
# Save results #
################

print('Saving Training Results')
test_train_history = h5.File(options['outPath']+"training_results.h5", 'w')
for stat in range(len(class_reg_trainer.stat_name)):
    if not class_reg_trainer.stat_name[stat] in options['print_metrics']: continue
    for split in range(len(split_name)):
        for timescale in range(len(timescale_name)):
            test_train_history.create_dataset(class_reg_trainer.stat_name[stat]+"_"+split_name[split]+"_"+timescale_name[timescale], data=np.array(class_reg_trainer.history[stat][split][timescale]))
if options['saveFinalModel']: 
    # save the the state_dicts instead of entire models. 
    torch.save(class_reg_trainer.model.net.state_dict(), options['outPath']+"saved_classifier.pt")
    if discriminator != None: torch.save(discriminator.net.state_dict(), options['outPath']+"saved_discriminator.pt")
    if generator != None: torch.save(generator.net.state_dict(), options['outPath']+"saved_generator.pt")

print('Getting Validation Results')
final_val_results = {}
class_reg_trainer.reset()
for sample in validationLoader:
    sample_results = class_reg_trainer.class_reg_eval(sample, store_reg_results=True)
    for key,data in sample_results.items():
        # cat together numpy array outputs
        if 'array' in str(type(data)):
            if key in final_val_results:
                final_val_results[key] = np.concatenate([final_val_results[key], data], axis=0)
            else:
                final_val_results[key] = data
        # put scalar outputs into a list
        else:
            final_val_results.setdefault(key, []).append(sample_results[key])

print('Saving Validation Results')
if len(options['val_outputs']) > 0:
    val_file = h5.File(options['outPath']+"validation_results.h5", 'w')
    for key,data in final_val_results.items():
        if not key in options['val_outputs']: continue
        val_file.create_dataset(key,data=np.asarray(data))
val_file.close()

##########################
# Analyze and make plots #
##########################

print('Making Plots')
analyzer.analyze([class_reg_trainer.model, discriminator, generator], test_train_history, final_val_results)
test_train_history.close()

end = timer()
print('Total time taken: %.2f minutes'%(float(end - start)/60.))
