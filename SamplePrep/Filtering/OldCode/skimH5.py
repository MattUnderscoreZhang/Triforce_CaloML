import h5py
import h5pp
import numpy as np
from os import listdir
from os.path import isfile, join
import sys, getopt


helpline = 'skimH5.py -i <inputPath> -o <outputPath>'


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, "hi:o:", ["inputPath=", "outputPath="])
    except getopt.GetoptError:
        print helpline
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpline
            sys.exit()
        elif opt in ("-i", "--inputPath"):
            inputPath = arg
        elif opt in ("-o", "--outputPath"):
            outputPath = arg

    print 'Converting all files found in ', inputPath
    print 'Saving converted files in ', outputPath

    # loop over all the files in the inputPath folder
    for fileName in [file for file in listdir(inputPath) if isfile(join(inputPath, file))]:
        print "Skimming", fileName

        oldFile = h5py.File(inputPath + fileName, "r")
        newFile = h5py.File(outputPath + fileName, "w")

        badIndices = [index for index, (ratio, ecal_e) in enumerate(zip(oldFile['HCAL_ECAL_ERatio'], oldFile["ECAL_E"])) if ratio > 0.025 or ecal_e == 0]

        # recursively list all datasets in the sample
        datasets = []

        def appendName(name):
            if isinstance(oldFile[name], h5py.Dataset):
                datasets.append(name)
            return None

        oldFile.visit(appendName)

        if len(badIndices) < len(oldFile['HCAL_ECAL_ERatio']):
            for dsetName in datasets:
                dset = h5pp.deleteRows(oldFile, dsetName, badIndices, newFile)

        oldFile.close()
        newFile.close()
