import os, sys
options = {}

##################
# Choose samples #
##################

basePath = "/data/shared/LCDLargeWindow/fixedangle/"
options['samplePath'] = [basePath + "Pi0Escan*/Pi0Escan_*.h5", basePath + "GammaEscan*/GammaEscan_*.h5"]
options['classPdgID'] = [111, 22] # [Pi0, Gamma]
# options['samplePath'] = [basePath + "ChPiEscan_1_MERGED/ChPiEscan_*.h5", basePath + "EleEscan_1_MERGED/EleEscan_*.h5"]
# options['classPdgID'] = [211, 11] # [ChPi, Ele]
options['eventsPerFile'] = 10000

###############
# Job options #
###############

options['importGPU'] = True

options['trainRatio'] = 0.90
options['relativeDeltaLossThreshold'] = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
options['relativeDeltaLossNumber'] = 5 # ...for this number of test losses in a row
options['earlyStopping'] = False
options['batchSize'] = 200 # 1000
#options['batchSize'] = 10 # 1000
options['saveFinalModel'] = 1 # takes a lot of space
options['saveModelEveryNEpochs'] = 0 # 0 to only save at end
options['outPath'] = os.getcwd()+"/Output/"+sys.argv[1]+"/"

options['nEpochs'] = 5 # break after this number of epochs
options['nTrainMax'] = -1
options['nTestMax'] = -1
options['nValidationMax'] = -1

# options['nEpochs'] = 1 # break after this number of epochs
# options['nTrainMax'] = 20
# options['nTestMax'] = 20
# options['nValidationMax'] = 20

## relative weight to assign to each type of output
## set to 0 to ignore a type of output
#options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 0.0, 'eta_regression': 0.0}
options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 1.0, 'eta_regression': 0.0}

################
# Choose tools #
################

from Architectures import Combined_Classifier, Combined_DNN, Discriminator, Generator
from Analysis import Classification_Plotter

_decayRate = 0
_nHiddenLayers = int(sys.argv[2])
_hiddenLayerNeurons = int(sys.argv[3])
_learningRate = float(sys.argv[4])
_dropoutProb = float(sys.argv[5])
_windowSizeECAL = int(sys.argv[6])
_windowSizeHCAL = int(sys.argv[7])

combined_classifier = Combined_DNN.Net(classes=options['classPdgID'], hiddenLayerNeurons=_hiddenLayerNeurons, nHiddenLayers=_nHiddenLayers, dropoutProb=_dropoutProb, learningRate=_learningRate, decayRate=_decayRate, windowSizeECAL=_windowSizeECAL, windowSizeHCAL=_windowSizeHCAL)
discriminator = None
generator = None
analyzer = Classification_Plotter.Analyzer()
