import os, sys

##################
# Choose samples #
##################

# basePath = "/data/LCD/V3/Original/EleChPi/"
# samplePath = [basePath + "ChPi/ChPiEscan_*.h5", basePath + "Ele/EleEscan_*.h5"]
# classPdgID = [211, 11] # absolute IDs corresponding to paths above
basePath = "/data/LCD/V3/Original/GammaPi0/"
samplePath = [basePath + "Pi0/Pi0Escan_*.h5", basePath + "Gamma/GammaEscan_*.h5"]
classPdgID = [111, 22] # absolute IDs corresponding to paths above
eventsPerFile = 10000
ECAL_size = (25, 25, 25)
HCAL_size = (5, 5, 60)

nworkers = 0 # number of workers in PyTorch DataLoader

###############
# Job options #
###############

trainRatio = 0.66
nEpochs = 5 # break after this number of epochs
relativeDeltaLossThreshold = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
relativeDeltaLossNumber = 5 # ...for this number of test losses in a row
batchSize = 200 # 1000
saveModelEveryNEpochs = 0 # 0 to only save at end

OutPath = os.getcwd()+"/Output/"+sys.argv[1]+"/"

################
# Choose tools #
################

from Classification import NIPS_Classifier
from Regression import NIPS_Regressor
from GAN import NIPS_GAN
from Analysis import Default_Analyzer

# _learningRate = float(sys.argv[2])
# _decayRate = float(sys.argv[3])
# _dropoutProb = float(sys.argv[4])
# _hiddenLayerNeurons = int(sys.argv[5])
# _nHiddenLayers = int(sys.argv[6])
_learningRate = 0.000001
_decayRate = 0
_dropoutProb = 0
_hiddenLayerNeurons = None
_nHiddenLayers = None

classifier = GoogLeNet.Classifier(_learningRate, _decayRate)
# classifier = NIPS_Classifier.Classifier(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
# classifier = None # set a tool to None to ignore it
regressor = NIPS_Regressor.Regressor(_learningRate, _decayRate)
GAN = NIPS_GAN.GAN(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
analyzer = Default_Analyzer.Analyzer()
