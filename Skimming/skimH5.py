import h5py
import h5pp
import numpy as np

oldFilePath = "../AllFiles/H5Files/v1/Unskimmed_withOpeningAngle_FixedCoordinates/"
newFilePath = "../AllFiles/H5Files/v1/Skimmed_withOpeningAngle_FixedCoordinates/"

filesToSkim = []

# for fileN in range(1, 11):
for fileN in range(1, 2):
    # filesToSkim.append("gamma_60_GeV_"+str(fileN)+".h5")
    filesToSkim.append("pi0_60_GeV_"+str(fileN)+".h5")

for fileName in filesToSkim:

    print "Skimming", fileName

    oldFile = h5py.File(oldFilePath + fileName)
    newFile = h5py.File(newFilePath + fileName, "w")

    badIndices = [index for index, ratio in enumerate(oldFile['HCAL_ECAL_Ratios/HCAL_ECAL_ERatio']) if ratio > 0.2 or not np.isfinite(ratio)]

    # recursively list all datasets in the sample
    datasets = []

    def appendName(name):
        if isinstance(oldFile[name], h5py.Dataset):
            datasets.append(name)
        return None

    oldFile.visit(appendName)

    if len(badIndices) < len(oldFile['HCAL_ECAL_Ratios/HCAL_ECAL_ERatio']):
        for dsetName in datasets:
            dset = h5pp.deleteRows(oldFile, dsetName, badIndices, newFile)

    oldFile.close()
    newFile.close()
