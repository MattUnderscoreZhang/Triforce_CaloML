from __future__ import division
import numpy as np
from skimage.util.shape import view_as_windows
import sys
import h5py as h5
import pdb

############################
# File reading and writing #
############################

def convertFile(inFile, outFile):

    # open file and extract events
    oldFile = h5.File(inFile)
    newFile = h5.File(outFile, "w")
    ECAL = oldFile["ECAL"][()]

    # copy over all other existing arrays from the old file
    for key in oldFile.keys():
        if key in ['ECAL','HCAL']: continue
        newFile.create_dataset(key, data=oldFile[key][()], compression='gzip')
    oldFile.close()

    ########################
    # Calculating features #
    ########################

    ECAL_z = np.sum(ECAL, axis=3) # sum in z
    R9 = [view_as_windows(event, (3,3)).sum(axis=(-2,-1)).max()/event.sum() for event in ECAL_z]

    # save features to h5 file
    newFile.create_dataset("R9", data=np.array(R9), compression='gzip')
    newFile.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    convertFile(inFile, outFile)
