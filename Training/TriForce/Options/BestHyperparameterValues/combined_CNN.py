import os, sys
options = {}

##################
# Choose samples #
##################

basePath = "/data/LCD/NewSamples/RandomAngle/CLIC/"
options['samplePath'] = [basePath + "ChPiEscan_RandomAngle_Filtered_2_Combined/ChPiEscan_*.h5", basePath + "EleEscan_RandomAngle_MERGED/EleEscan_*.h5"]
options['classPdgID'] = [211, 11] # [ChPi, Ele]

###############
# Job options #
###############

options['trainRatio'] = 0.90
options['relativeDeltaLossThreshold'] = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
options['relativeDeltaLossNumber'] = 5 # ...for this number of test losses in a row
options['earlyStopping'] = False
# options['microBatchSize'] = 10 # number of events at a time on the GPU
# options['nMicroBatchesInMiniBatch'] = 100 # total # in minibatch = nMiniBatches * batchSize
options['microBatchSize'] = 1000 # number of events at a time on the GPU
options['nMicroBatchesInMiniBatch'] = 1 # total # in minibatch = nMiniBatches * batchSize
options['saveFinalModel'] = 0 # takes a lot of space
options['saveModelEveryNEpochs'] = 0 # 0 to only save at end
options['outPath'] = os.getcwd()+"/Output/"+sys.argv[1]+"/"

options['nEpochs'] = 100 # break after this number of epochs
options['nTrainMax'] = 20
options['nTestMax'] = 20
options['nValidationMax'] = 1

options['skipClassRegTrain'] = False # skips training class+reg network and loads already trained net instead. 

# options['nEpochs'] = 1 # break after this number of epochs
# options['nTrainMax'] = 20
# options['nTestMax'] = 20
# options['nValidationMax'] = 20

## relative weight to assign to each type of output
## set to 0 to ignore a type of output
options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 1.0, 'eta_regression': 10.0, 'phi_regression': 0.0}
# options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 200.0, 'eta_regression': 0.0, 'phi_regression': 0.0}
# options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 200.0, 'eta_regression': 500.0, 'phi_regression': 100.0}

#################
# Input filters #
#################

from Loader import filters

#energy_filter = filters.energy_filter(50,70)
#options['filters'] = [energy_filter]
options['filters'] = []

## filter for variable angle ele, gamma, pi0
# hOverE_filter = filters.hOverE_filter(0.4)
# recoOverGen_filter = filters.recoOverGen_filter(0.66)
# options['filters'] = [hOverE_filter, recoOverGen_filter]

##################
# Output options #
##################

# options['print_metrics'] = ['class_reg_loss', 'class_acc']
options['print_metrics'] = ['class_reg_loss', 'class_loss', 'reg_energy_loss', 'reg_eta_loss', 'reg_phi_loss', 'class_acc', 'class_sig_acc', 'class_bkg_acc', 'reg_energy_bias', 'reg_energy_res', 'reg_eta_diff', 'reg_eta_std', 'reg_phi_diff', 'reg_phi_std']

# options['val_outputs'] = []
options['val_outputs'] = ['energy', 'eta', 'phi', 'recoEta', 'recoPhi', 'ECAL_E', 'HCAL_E', 'pdgID', 'class_raw_prediction', 'class_prediction', 'class_truth', 'reg_energy_prediction', 'reg_eta_prediction', 'reg_phi_prediction']

################
# Choose tools #
################

from Architectures import Combined_DNN, Combined_CNN, GoogLeNet, Discriminator, Generator
from Analysis import Plotter

options['decayRate'] = 0
options['nHiddenLayers'] = 4
options['hiddenLayerNeurons'] = 512
options['learningRate'] = 0.0004
options['dropoutProb'] = 0.12
options['windowSizeECAL'] = 25
options['windowSizeHCAL'] = 11

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

# scaling to apply to input values in nets
options['inputScaleSumE'] = 0.01
options['inputScaleEta'] = 10.0
options['inputScalePhi'] = 10.0

# ADD GAN OPTIONS
# combined_classifier = GoogLeNet.Net(options)
combined_classifier = Combined_CNN.Net(options)
discriminator = Discriminator.Net(options)
generator = Generator.Net(options)
analyzer = Plotter.Analyzer()
