import h5py
import h5pp
import numpy as np

oldFilePath = "H5Files/v1/Unskimmed/"
newFilePath = "H5Files/v1/Skimmed/"

for fileN in range(1, 11):
    filesToSkim.append("gamma_60_GeV_"+str(fileN)+".h5")
    filesToSkim.append("pi0_60_GeV_"+str(fileN)+".h5")

for fileName in filesToSkim:

    print "Skimming", fileName

    oldFile = h5py.File(oldFilePath + fileName)
    newFile = h5py.File(newFilePath + fileName, "w")

    badIndices = [index for index, ratio in enumerate(oldFile['ECAL_HCAL_ERatio']) if ratio < 5 or not np.isfinite(ratio)]

    if len(badIndices) < len(oldFile['ECAL_HCAL_ERatio']):
        for dsetName in oldFile.keys():
            dset = h5pp.deleteRows(oldFile, dsetName, badIndices, newFile)

    oldFile.close()
    newFile.close()
