from __future__ import division
import numpy as np
import sys
import h5py
import os
import collections
import glob

def normalizeFiles(inFile, outFile):

    files=sorted(glob.glob(inFile))
    totalFeatures = collections.OrderedDict()
    means = collections.OrderedDict()
    stds = collections.OrderedDict()

    for file_index in range(len(files)):
        f1 = h5py.File(files[file_index], 'r')
        for feature in f1.keys():
            if feature in ['ECAL', 'HCAL'] pass
            if file_index == 0:
                totalFeatures[feature] = f1[feature]
            else:
                totalFeatures[feature] = np.concatentate(totalFeatures[feature], f1[feature])
        f1.close()

    for feature in totalFeatures.keys():
        means[feature] = np.mean(totalFeatures[feature])
        stds[feature] = np.std(totalFeatures[feature])

    for file_index in range(len(files)):
        # Save features to an h5 file
        f1 = h5py.File(outFile+os.path.basename(files[file_index]), "w")
        for feature in f1.keys():
            if feature in ['ECAL', 'HCAL'] pass
            f1[feature] = (totalFeatures[feature] - means[feature])/stds[feature]
        f1.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    pdgID = float(sys.argv[3])
    convertFile(inFile, outFile, pdgID)
