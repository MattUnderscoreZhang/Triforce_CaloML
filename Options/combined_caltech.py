import os, sys
options = {}

##################
# Choose samples #
##################

basePath = "/data/shared/LCDLargeWindow/fixedangle/"
options['samplePath'] = [basePath + "ChPiEscan/temp1/*.h5", basePath + "EleEscan/temp1/*.h5"]
# options['samplePath'] = [basePath + "ChPiEscan/temp/split/1_1/*.h5", basePath + "EleEscan/temp/split/1_1/*.h5"]
# options['classPdgID'] = [111, 22] # [Pi0, Gamma]
# options['samplePath'] = [basePath + "ChPiEscan_1_MERGED/ChPiEscan_*.h5", basePath + "EleEscan_1_MERGED/EleEscan_*.h5"]
options['classPdgID'] = [211, 11] # [ChPi, Ele]
options['eventsPerFile'] = 200 #1000

###############
# Job options #
###############
options['importGPU'] = True 
options['nWorkers'] = 71 # has to be > 2
options['nLoaders'] = 50 # has to be less than nWorkers by at least 2. 
options['trainRatio'] = 1
options['validationRatio'] = 0
options['relativeDeltaLossThreshold'] = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
options['relativeDeltaLossNumber'] = 5 # ...for this number of test losses in a row
options['earlyStopping'] = False
options['batchSize'] = 1000
options['saveFinalModel'] = 1 # takes a lot of space
options['saveModelEveryNEpochs'] = 0 # 0 to only save at end
options['outPath'] = os.getcwd()+"/Output/"+"/"
# sys.argv[1]+
options['nEpochs'] = 1 # break after this number of epochs
options['nTrainMax'] = -1
options['nTestMax'] = -1
options['nValidationMax'] = -1

# options['nEpochs'] = 1 # break after this number of epochs
# options['nTrainMax'] = 1
# options['nTestMax'] = 1
# options['nValidationMax'] = 1

## relative weight to assign to each type of output
## set to 0 to ignore a type of output
#options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 0.0, 'eta_regression': 0.0}
options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 1.0, 'eta_regression': 0.0}

#################
# Input filters #
#################

from Loader import filters

#energy_filter = filters.energy_filter(50,70)
#hOverE_filter = filters.hOverE_filter(0.4)
#options['filters'] = [hOverE_filter]
options['filters'] = []

##################
# Output options #
##################

options['print_metrics'] = ['class_reg_loss', 'class_loss', 'reg_energy_loss', 'reg_eta_loss', 'class_acc', 'class_sig_acc', 'class_bkg_acc', 'reg_energy_bias', 'reg_energy_res', 'reg_eta_diff', 'reg_eta_std']

options['val_outputs'] = ['energy', 'eta', 'phi', 'recoEta', 'recoPhi', 'ECAL_E', 'HCAL_E', 'pdgID', 'reg_energy_prediction', 'reg_eta_prediction']

################
# Choose tools #
################

from Architectures import Combined_DNN, Combined_CNN, Discriminator, Generator
from Analysis import Plotter


options['decayRate'] = 0
options['nHiddenLayers'] = 5 #int(sys.argv[2])
options['hiddenLayerNeurons'] = 1024 #int(sys.argv[3])
options['learningRate'] = 0.01 #float(sys.argv[4])
options['dropoutProb'] = 0.5 #float(sys.argv[5])
options['windowSizeECAL'] = 51 #int(sys.argv[6])
options['windowSizeHCAL'] = 11 #int(sys.argv[7])

# options specific to CNN
# can make these also configurable from command line
options['nfiltECAL'] = 3
options['kernelxyECAL'] = 4
options['kernelzECAL'] = 4
options['nfiltHCAL'] = 3
options['kernelxyHCAL'] = 2
options['kernelzHCAL'] = 6
options['maxpoolkernelECAL'] = 2
options['maxpoolkernelHCAL'] = 2

#combined_classifier = Combined_DNN.Net(options)
combined_classifier = Combined_CNN.Net(options)
discriminator = None
generator = None
analyzer = Plotter.Analyzer()
