from Loader import loader
import glob

basePath = "/data/LCD/NewSamples/Fixed/"
options = {}
options['samplePath'] = [basePath + "Pi0Escan*/Pi0Escan_*.h5", basePath + "GammaEscan*/GammaEscan_*.h5"]
options['classPdgID'] = [111, 22] # [Pi0, Gamma]
options['microBatchSize'] = 1000 # number of events at a time on the GPU
options['nLoaders'] = 8
options['filters'] = []

nClasses = len(options['samplePath'])
classFiles = [[]] * nClasses
for i, classPath in enumerate(options['samplePath']):
    classFiles[i] = glob.glob(classPath)
trainFiles = []
for i in range(10):
    newFiles = []
    for j in range(nClasses):
        newFiles.append(classFiles[j][i])
    trainFiles.append(newFiles)

trainLoader = loader.HDF5Dataset(trainFiles, options['classPdgID'], batch_size=options['microBatchSize'], n_workers=options['nLoaders'], filters=options['filters'])
for data in trainLoader:
    pass
