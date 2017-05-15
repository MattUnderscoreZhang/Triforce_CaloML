import h5py
import h5pp
import numpy as np

oldFilePath = "H5Files/"
newFilePath = "SkimmedH5Files/"
filesToSkim = ["gamma_60_GeV.h5"]

for fileName in filesToSkim:

    oldFile = h5py.File(oldFilePath + fileName)
    newFile = h5py.File(newFilePath + fileName, "w")

    badIndices = [index for index, ratio in enumerate(oldFile['ECAL_HCAL_ERatio']) if ratio < 5 or not np.isfinite(ratio)]

    if len(badIndices) < len(oldFile['ECAL_HCAL_ERatio']):
        for dsetName in oldFile.keys():
            dset = h5pp.deleteRows(oldFile, dsetName, badIndices, newFile)

    oldFile.close()
    newFile.close()
