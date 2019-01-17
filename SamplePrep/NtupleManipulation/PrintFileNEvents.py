# see how many events are in each file

import glob
import h5py as h5

basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/"
samplePath = [basePath + "Pi0Escan_*_MERGED/Pi0Escan_*.h5", basePath + "GammaEscan_*_MERGED/GammaEscan_*.h5"]
classFiles = []
for i, classPath in enumerate(samplePath):
    classFiles += glob.glob(classPath)

for file in classFiles:
    f = h5.File(file)
    print(f, "has", f['ECAL'].shape[0], "events")
