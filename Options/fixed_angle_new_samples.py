import os, sys
options = {}

##################
# Choose samples #
##################

basePath = "/data/LCD/NewSamples/FixedEnergyFilter/"
options['samplePath'] = [basePath + "Pi0Escan*/Pi0Escan_*.h5", basePath + "GammaEscan*/GammaEscan_*.h5"]
options['classPdgID'] = [111, 22] # [Pi0, Gamma]
# options['samplePath'] = [basePath + "ChPiEscan_1_MERGED/ChPiEscan_*.h5", basePath + "EleEscan_1_MERGED/EleEscan_*.h5"]
# options['classPdgID'] = [211, 11] # [ChPi, Ele]
options['eventsPerFile'] = 1000

###############
# Job options #
###############

options['trainRatio'] = 0.90
options['relativeDeltaLossThreshold'] = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
options['relativeDeltaLossNumber'] = 5 # ...for this number of test losses in a row
options['earlyStopping'] = False
options['batchSize'] = 200 # 1000
options['saveFinalModel'] = 1 # takes a lot of space
options['saveModelEveryNEpochs'] = 0 # 0 to only save at end
options['outPath'] = os.getcwd()+"/Output/"+sys.argv[1]+"/"

options['nEpochs'] = 20 # break after this number of epochs
options['nTrainMax'] = -1
options['nTestMax'] = -1
options['nValidationMax'] = -1

options['lossTermWeights'] = {'classification': 1.0, 'energy_regression': 0.0, 'eta_regression': 0.0}
options['filters'] = []

# options['nEpochs'] = 1 # break after this number of epochs
# options['nTrainMax'] = 20
# options['nTestMax'] = 20
# options['nValidationMax'] = 20

# options['print_metrics'] = ['class_reg_loss', 'class_acc']
options['print_metrics'] = ['class_reg_loss', 'class_acc', 'class_sig_acc', 'class_bkg_acc']

# # options['val_outputs'] = []
# options['val_outputs'] = ['reg_energy_truth', 'reg_energy_prediction', 'reg_eta_truth', 'reg_eta_prediction', 'reg_raw_ECAL_E', 'reg_raw_HCAL_E']

################
# Choose tools #
################

from Architectures import Fixed_Angle_Classifier
# from Classification import GoogLeNet
# from Regression import NIPS_Regressor
# from GAN import NIPS_GAN
from Analysis import Classification_Plotter

# options['decayRate'] = 0
# options['nHiddenLayers'] = int(sys.argv[2])
# options['hiddenLayerNeurons'] = int(sys.argv[3])
# options['learningRate'] = float(sys.argv[4])
# options['dropoutProb'] = float(sys.argv[5])
# options['windowSizeECAL'] = int(sys.argv[6])
# options['windowSizeHCAL'] = 11

_decayRate = 0
_nHiddenLayers = int(sys.argv[2])
_hiddenLayerNeurons = int(sys.argv[3])
_learningRate = float(sys.argv[4])
_dropoutProb = float(sys.argv[5])
_windowSize = int(sys.argv[6])

# classifier = GoogLeNet.Classifier(_learningRate, _decayRate)
classifier = Fixed_Angle_Classifier.Classifier(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate, _windowSize)
regressor = None # NIPS_Regressor.Regressor(_learningRate, _decayRate)
generator = None
# analyzer = Default_Analyzer.Analyzer()
analyzer = Classification_Plotter.Analyzer()
