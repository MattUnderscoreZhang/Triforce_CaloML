import os, sys, yaml

with open(os.path.join(os.getcwd(),'cfgs/GoogLeNet.yaml')) as f: 
    cfgs = yaml.load(f)

##################
# Choose samples #
##################

# basePath = "/data/LCD/V3/Original/EleChPi/"
# samplePath = [basePath + "ChPi/ChPiEscan_*.h5", basePath + "Ele/EleEscan_*.h5"]
# classPdgID = [211, 11] # absolute IDs corresponding to paths above
try: 
    basePath = cfgs['DATA']['basePath']
except: 
    print("Missing Param: No parameter in yaml file named 'basePath'. ")
try: 
    samplePath = cfgs['DATA']['samplePath']
except: 
    print("Missing Param: No parameter in yaml file named 'samplePath'. ")
try: 
    classPdgID = cfgs['DATA']['classPdgID'] # absolute IDs corresponding to paths above
except: 
    print("Missing Param: No parameter in yaml file named 'classPdgID'. ")
try: 
    eventsPerFile = cfgs['DATA']['eventsPerFile']
except: 
    print("Missing Param: No parameter in yaml file named 'eventsPerFile'. ")

try: 
    nworkers = 0 # number of workers in PyTorch DataLoader
except: 
    print("Missing Param: No parameter in yaml file named 'nworkers'. ")

###############
# Job options #
###############
for category in cfgs['TRAIN'].keys(): 
    for module in cfgs['TRAIN'][category].keys(): 
        try: 
            trainRatio = cfgs['TRAIN'][category][module]['trainRatio']
        except: 
            print("Missing Param: No parameter in yaml file named 'trainRatio'. ")
        try: 
            nEpochs = cfgs['TRAIN'][category][module]['nEpochs'] # break after this number of epochs
        except: 
            print("Missing Param: No parameter in yaml file named 'nEpochs'. ")
        try: 
            relativeDeltaLossThreshold = cfgs['TRAIN'][category][module]['relativeDeltaLossThreshold'] # break if change in loss falls below this threshold over an entire epoch, or...
        except:
            print("Missing Param: No parameter in yaml file named 'relativeDeltaLossThreshold'. ")
        try: 
            relativeDeltaLossNumber = cfgs['TRAIN'][category][module]['relativeDeltaLossNumber'] # ...for this number of test losses in a row
        except: 
            print("Missing Param: No parameter in yaml file named 'relativeDeltaLossNumber'. ")
        try: 
            batchSize = cfgs['TRAIN'][category][module]['batchSize'] # 1000
        except: 
            print("Missing Param: No parameter in yaml file named 'batchSize'. ")


        OutPath = os.getcwd()+"/Output/"+sys.argv[1]+"/"


        ################
        # Choose tools #
        ################

        from Classification import GoogLeNet # NIPS_Classifier
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

<<<<<<< HEAD
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
=======
        classifier = GoogLeNet.Classifier(_learningRate, _decayRate)
        # classifier = NIPS_Classifier.Classifier(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
        # classifier = None # set a tool to None to ignore it
        regressor = None
        # regressor = NIPS_Regressor.Regressor(_learningRate, _decayRate)
        GAN = None
        # GAN = NIPS_GAN.GAN(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
        analyzer = Default_Analyzer.Analyzer()
>>>>>>> junze_dev
