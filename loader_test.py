from Loader import loader
import glob, sys
import threading
from timeit import default_timer as timer

###########
# Options #
###########

from Options.loader_test import *

##############
# Load files #
##############

# if these parameters are not set, give them default values
defaultParameters = {'importGPU':False, 'nTrainMax':-1, 'nValidationMax':-1, 'nTestMax':-1, 'validationRatio':0, 'nWorkers':0, 'calculate_loss_per_n_batches':20, 'test_loss_eval_max_n_batches':10, 'earlyStopping':False, 'relativeDeltaLossThreshold':0, 'relativeDeltaLossNumber':5, 'saveModelEveryNEpochs':0, 'saveFinalModel':0, 'train_class_reg_separately':False}
for optionName in defaultParameters.keys():
    if optionName not in options.keys():
        options[optionName] = defaultParameters[optionName]

# gather sample files for each type of particle
nClasses = len(options['samplePath'])
classFiles = [[]] * nClasses
for i, classPath in enumerate(options['samplePath']):
    classFiles[i] = glob.glob(classPath)

# determine numbers of test, train, and validation files
print('-------------------------------')
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
trainLoader = loader.ThreadedLoader(trainFiles, options['batchSize'], options['classPdgID'], options['filters'])
validationLoader = loader.ThreadedLoader(validationFiles, options['batchSize'], options['classPdgID'], options['filters'])
testLoader = loader.ThreadedLoader(testFiles, options['batchSize'], options['classPdgID'], options['filters'])

########################
# File reading threads #
########################

trainLoader = loader.ThreadedLoader(trainFiles, options['batchSize'], options['classPdgID'], options['filters'])
start = timer()
for data in trainLoader:
    print(data)
    if data[1] >= 10: break
end = timer()
print(end-start)

trainLoader = loader.ThreadedLoader(trainFiles, options['batchSize'], options['classPdgID'], options['filters'])
start = timer()
class LoaderThread (threading.Thread):
    def __init__(self, loader):
        threading.Thread.__init__(self)
        self.loader = loader
    def run(self):
        for data in self.loader:
            print(data)
            if data[1] >= 10:
                end = timer()
                print(end-start)
                break

n_workers = 4
for _ in range(n_workers):
    t = LoaderThread(trainLoader)
    t.start()

# def threaded_batches_feeder(batches_queue, loader):
    # for batch, data in enumerate(loader):
        # batches_queue.put((batch, data), block=True)

# for epoch in range(num_epoches):
    # for batch in range(batches_per_epoch):
	# #We fetch a GPU batch in 0's due to the queue mechanism
	# _, (batch_images, batch_labels) = cuda_batches_queue.get(block=True)
	# #train batch is the method for your training step.
	# #no need to pin_memory due to diminished cuda transfers using queues.
	# loss, accuracy = train_batch(batch_images, batch_labels)

# train_thread_killer.toggle_kill(True)
# cuda_transfers_thread_killer.toggle_kill(True)
# for _ in range(preprocess_workers):
    # try:
	# #Enforcing thread shutdown
	# train_batches_queue.get(block=True,timeout=1)
	      # cuda_batches_queue.get(block=True,timeout=1)
    # except Empty:
	# pass

####################
# Check generators #
####################

# for data in train_batches_queue.get(block=True):
    # print("BLAH")
