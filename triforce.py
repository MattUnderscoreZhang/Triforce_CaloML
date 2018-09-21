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
import numpy as np
import h5py as h5
import Loader.loader as loader
from Loader import transforms
import Options
sys.dont_write_bytecode = True # prevent the creation of .pyc files
import pdb
from timeit import default_timer as timer

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
history = historyData()

# enumerate parts of the data structure
stat_name = ['class_reg_loss', 'class_loss', 'reg_energy_loss', 'reg_eta_loss', 'reg_phi_loss', 'class_acc', 'class_sig_acc', 'class_bkg_acc', 'reg_energy_bias', 'reg_energy_res', 'reg_eta_diff', 'reg_eta_std', 'reg_phi_diff', 'reg_phi_std']
# stat metrics to print out every N batches
print_metrics = options['print_metrics']
CLASS_LOSS, CLASS_ACC = 0, 1
split_name = ['train', 'validation', 'test']
TRAIN, VALIDATION, TEST = 0, 1, 2
timescale_name = ['batch', 'epoch']
BATCH, EPOCH = 0, 1

#####################################
# Training and Evaluation Functions #
#####################################

class Trainer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.microbatch_n = 0
        combined_classifier.optimizer.zero_grad()

    def sgl_bkgd_acc(self, predicted, truth): 
        truth_sgl = truth.nonzero() # indices of non-zero elements in truth
        truth_bkgd = (truth == 0).nonzero() # indices of zero elements in truth
        correct_sgl_frac = 0
        correct_bkgd_frac = 0
        if len(truth_sgl) > 0:
            correct_sgl = 0
            for i in range(truth_sgl.shape[0]):
                if predicted[truth_sgl[i]][0] == truth[truth_sgl[i]][0]:
                    correct_sgl += 1
            correct_sgl_frac = float(correct_sgl / truth_sgl.shape[0])
        if len(truth_bkgd) > 0:
            correct_bkgd = 0
            for i in range(truth_bkgd.shape[0]):
                if predicted[truth_bkgd[i]][0] == truth[truth_bkgd[i]][0]:
                    correct_bkgd += 1
            correct_bkgd_frac = float(correct_bkgd / truth_bkgd.shape[0])
        return correct_sgl_frac, correct_bkgd_frac # signal acc, bkg acc

    def class_reg_eval(self,event_data, do_training=False, store_reg_results=False):
        if do_training:
            combined_classifier.net.train()
        else:
            combined_classifier.net.eval()
        outputs = combined_classifier.net(event_data)
        return_event_data = {}
        # classification
        truth_class = Variable(event_data["classID"].cuda())
        class_reg_loss = combined_classifier.lossFunction(outputs, event_data, options['lossTermWeights'])
        if do_training:
            microbatch_norm_loss = {}
            for key in class_reg_loss:
                microbatch_norm_loss[key] = class_reg_loss[key] / options['nMicroBatchesInMiniBatch']
            microbatch_norm_loss["total"].backward()
            self.microbatch_n += 1
            if (self.microbatch_n >= options['nMicroBatchesInMiniBatch']):
                self.microbatch_n = 0
                combined_classifier.optimizer.step()
        _, predicted_class = torch.max(outputs['classification'], 1) # max index in each event
        class_sig_acc, class_bkg_acc = self.sgl_bkgd_acc(predicted_class.data, truth_class.data)
        # regression outputs. move first to cpu
        pred_energy = transforms.pred_energy_from_reg(outputs['energy_regression'].data.cpu(), event_data)
        truth_energy = event_data["energy"]
        reldiff_energy = 100.0*(truth_energy - pred_energy)/truth_energy
        pred_eta = transforms.pred_eta_from_reg(outputs['eta_regression'].data.cpu(), event_data)
        diff_eta = event_data["eta"] - pred_eta
        pred_phi = transforms.pred_phi_from_reg(outputs['phi_regression'].data.cpu(), event_data)
        diff_phi = event_data["phi"] - pred_phi
        # return values
        return_event_data["class_reg_loss"] = class_reg_loss["total"].item()
        return_event_data["class_loss"] = class_reg_loss["classification"].item()
        return_event_data["reg_energy_loss"] = class_reg_loss["energy"].item()
        return_event_data["reg_eta_loss"] = class_reg_loss["eta"].item()
        return_event_data["reg_phi_loss"] = class_reg_loss["phi"].item()
        return_event_data["class_acc"] = float((predicted_class.data == truth_class.data).sum())/truth_class.shape[0]
        return_event_data["class_raw_prediction"] = outputs['classification'].data.cpu().numpy()[:,1] # getting the second number for 2-class classification
        return_event_data["class_prediction"] = predicted_class.data.cpu().numpy()
        return_event_data["class_truth"] = truth_class.data.cpu().numpy()
        return_event_data["class_sig_acc"] = class_sig_acc
        return_event_data["class_bkg_acc"] = class_bkg_acc
        return_event_data["reg_energy_bias"] = torch.mean(reldiff_energy)
        return_event_data["reg_energy_res"] = torch.std(reldiff_energy)
        return_event_data["reg_eta_diff"] = torch.mean(diff_eta)
        return_event_data["reg_eta_std"] = torch.std(diff_eta)
        return_event_data["reg_phi_diff"] = torch.mean(diff_phi)
        return_event_data["reg_phi_std"] = torch.std(diff_phi)
        return_event_data["energy"] = event_data["energy"].numpy()
        return_event_data["eta"] = event_data["eta"].numpy()
        return_event_data["openingAngle"] = event_data["openingAngle"].numpy()
        if store_reg_results:
            return_event_data["reg_energy_prediction"] = pred_energy.numpy()
            return_event_data["reg_eta_prediction"] = pred_eta.numpy()
            return_event_data["reg_phi_prediction"] = pred_phi.numpy()
            ECAL = event_data["ECAL"]
            return_event_data["ECAL_E"] = torch.sum(ECAL.view(ECAL.shape[0], -1), dim=1).view(-1).numpy()
            HCAL = event_data["HCAL"]
            return_event_data["HCAL_E"] = torch.sum(HCAL.view(HCAL.shape[0], -1), dim=1).view(-1).numpy()
            return_event_data["pdgID"] = event_data["pdgID"].numpy()
            return_event_data["phi"] = event_data["phi"].numpy()
            return_event_data["recoEta"] = event_data["recoEta"].numpy()
            return_event_data["recoPhi"] = event_data["recoPhi"].numpy()
        return return_event_data

    def class_reg_train(self, event_data):
        return self.class_reg_eval(event_data, do_training=True)

trainer = Trainer()

############################################
# Perform Training for Combined Classifier #
############################################

def update_batch_history(data, train_or_test, minibatch_n):
    if train_or_test == TRAIN:
        eval_results = trainer.class_reg_train(data)
    else:
        eval_results = trainer.class_reg_eval(data)
    for stat in range(len(stat_name)):
        history[stat][train_or_test][BATCH][minibatch_n] += (eval_results[stat_name[stat]] / options['nMicroBatchesInMiniBatch'])

def update_epoch_history():
    for stat in range(len(stat_name)):
        for split in [TRAIN, TEST]:
            history[stat][split][EPOCH].append(history[stat][split][BATCH][-1])

def print_stats(timescale):
    print_prefix = "epoch " if timescale == EPOCH else ""
    for split in [TRAIN, TEST]:
        if timescale == EPOCH and split == TRAIN: continue
        print(print_prefix + split_name[split] + ' sample')
        for stat in range(len(stat_name)):
            if stat_name[stat] in print_metrics:
                print('  ' + stat_name[stat] + ':\t %8.4f' % (history[stat][split][timescale][-1]))
        print()

# early stopping
previous_total_test_loss = 0
previous_epoch_total_test_loss = 0
delta_loss_below_threshold_count = 0

def should_i_stop(timescale):

    if not options['earlyStopping']: return False

    total_test_loss = history[CLASS_LOSS][TEST][timescale][-1]

    if timescale == BATCH:
        relative_delta_loss = 1 if previous_total_test_loss==0 else (previous_total_test_loss - total_test_loss)/(previous_total_test_loss)
        previous_total_test_loss = total_test_loss
        if (relative_delta_loss < options['relativeDeltaLossThreshold']): delta_loss_below_threshold_count += 1
        if (delta_loss_below_threshold_count >= options['relativeDeltaLossNumber']): return True
        else: delta_loss_below_threshold_count = 0
    elif timescale == EPOCH:
        relative_delta_loss = 1 if previous_epoch_total_test_loss==0 else (previous_epoch_total_test_loss - epoch_total_test_loss)/(previous_epoch_total_test_loss)
        previous_epoch_total_test_loss = total_test_loss
        if (relative_delta_loss < options['relativeDeltaLossThreshold']): return True

    return False

def class_reg_training():

    train_data = None
    test_data = None
    minibatch_n = 0
    end_training = False

    for epoch in range(options['nEpochs']):

        train_or_test = TRAIN
        trainIter = iter(trainLoader)
        testIter = iter(testLoader)
        break_loop = False
        while True:
            trainer.reset()
            for _ in range(options['nMicroBatchesInMiniBatch']):
                try:
                    train_data = next(trainIter)
                    test_data = next(testIter)
                    if train_or_test == TRAIN:
                        update_batch_history(train_data, train_or_test, minibatch_n)
                    else:
                        update_batch_history(test_data, train_or_test, minibatch_n)
                except StopIteration:
                    break_loop = True
            if break_loop:
                break
            if train_or_test == TEST:
                print('-------------------------------')
                print('epoch %d, batch %d' % (epoch+1, minibatch_n))
                print_stats(BATCH)
                minibatch_n += 1
            if should_i_stop(BATCH): end_training = True
            if train_or_test == TEST:
                train_or_test = TRAIN
            else:
                train_or_test = TEST

        # end of epoch
        update_epoch_history()
        print('-------------------------------')
        print_stats(EPOCH)
        # plot every epoch
        analyzer.analyze_online(history, options['outPath'])
        if should_i_stop(EPOCH): end_training = True

        # save results
        # should these be state_dicts?
        if options['saveFinalModel'] and (options['saveModelEveryNEpochs'] > 0) and ((epoch+1) % options['saveModelEveryNEpochs'] == 0):
            if not os.path.exists(options['outPath']): os.makedirs(options['outPath'])
            torch.save(combined_classifier.net, options['outPath']+"saved_classifier_epoch_"+str(epoch)+".pt")
            if discriminator != None: torch.save(discriminator.net, options['outPath']+"saved_discriminator_epoch_"+str(epoch)+".pt")
            if generator != None: torch.save(generator.net, options['outPath']+"saved_generator_epoch_"+str(epoch)+".pt")

        if end_training: break

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
            class_reg_training() 
        else:
            sys.exit()
else: 
    print('-------------------------------')
    print('Training Classifier and Regressor')
    class_reg_training() 
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
for stat in range(len(stat_name)):
    if not stat_name[stat] in print_metrics: continue
    for split in range(len(split_name)):
        for timescale in range(len(timescale_name)):
            test_train_history.create_dataset(stat_name[stat]+"_"+split_name[split]+"_"+timescale_name[timescale], data=np.array(history[stat][split][timescale]))
if options['saveFinalModel']: 
    # save the the state_dicts instead of entire models. 
    torch.save(combined_classifier.net.state_dict(), options['outPath']+"saved_classifier.pt")
    if discriminator != None: torch.save(discriminator.net.state_dict(), options['outPath']+"saved_discriminator.pt")
    if generator != None: torch.save(generator.net.state_dict(), options['outPath']+"saved_generator.pt")

print('Getting Validation Results')
final_val_results = {}
trainer.reset()
for sample in validationLoader:
    sample_results = trainer.class_reg_eval(sample, store_reg_results=True)
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
analyzer.analyze([combined_classifier, discriminator, generator], test_train_history, final_val_results)
test_train_history.close()

end = timer()
print('Total time taken: %.2f minutes'%(float(end - start)/60.))
