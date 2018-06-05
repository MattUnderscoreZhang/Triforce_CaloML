import os, sys
options = {}

##################
# Choose samples #
##################

basePath = "/data/LCD/V3/Original/GammaPi0/"
# options['samplePath'] = [basePath + "ChPi/ChPiEscan_*.h5", basePath + "Ele/EleEscan_*.h5"]
# options['classPdgID'] = [211, 11] # absolute IDs corresponding to paths above
options['samplePath'] = [basePath + "Pi0/Pi0Escan_*.h5", basePath + "Gamma/GammaEscan_*.h5"]
options['classPdgID'] = [111, 22] # [Pi0, Gamma]
options['eventsPerFile'] = 10000

###############
# Job options #
###############

options['trainRatio'] = 0.66
options['relativeDeltaLossThreshold'] = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
options['relativeDeltaLossNumber'] = 5 # ...for this number of test losses in a row
options['batchSize'] = 200 # 1000
options['saveFinalModel'] = 1 # takes a lot of space
options['saveModelEveryNEpochs'] = 0 # 0 to only save at end
options['nEpochs'] = 10 # break after this number of epochs
options['outPath'] = os.getcwd()+"/Output/"+sys.argv[1]+"/"

################
# Choose tools #
################

from Classification import NIPS_Classifier
from Regression import NIPS_Regressor
from GAN import NIPS_GAN
from Analysis import Classification_Plotter

_learningRate = 0.000001 # 0.001 
_decayRate = 0
_dropoutProb = 0 # 0.5
_hiddenLayerNeurons = None # 256
_nHiddenLayers = None # 4

classifier = NIPS_Classifier.Classifier(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
regressor = None # NIPS_Regressor.Regressor(_learningRate, _decayRate)
generator = None
# analyzer = Default_Analyzer.Analyzer()
analyzer = Classification_Plotter.Analyzer()
