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

nworkers = 0 # number of workers in PyTorch DataLoader

###############
# Job options #
###############

trainRatio = 0.66
nEpochs = 5 # break after this number of epochs
relativeDeltaLossThreshold = 0.001 # break if change in loss falls below this threshold over an entire epoch, or...
relativeDeltaLossNumber = 5 # ...for this number of test losses in a row
batchSize = 1000

OutPath = os.getcwd()+"/Output/"+sys.argv[1]+"/"

################
# Choose tools #
################

from Classification import Default_Classifier
from Regression import Default_Regressor
from GAN import Default_GAN
from Analysis import Default_Analyzer

_learningRate = float(sys.argv[2])
_decayRate = float(sys.argv[3])
_dropoutProb = float(sys.argv[4])
_hiddenLayerNeurons = int(sys.argv[5])
_nHiddenLayers = int(sys.argv[6])

classifier = Default_Classifier.Classifier(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
regressor = None
GAN = None
analyzer = Default_Analyzer.Analyzer()
