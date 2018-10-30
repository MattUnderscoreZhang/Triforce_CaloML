from __future__ import division
import numpy as np
from skimage.util.shape import view_as_windows
import sys
import h5py as h5
import pdb

############################
# File reading and writing #
############################

def convertFile(inFile):

    # open file and extract events
    oldFile = h5.File(inFile)
    ECAL = oldFile["ECAL"][()]

    ########################
    # Calculating features #
    ########################

    ECAL_z = np.sum(ECAL, axis=3) # sum in z
    R9 = [view_as_windows(event, (3,3)).sum(axis=(-2,-1)).max()/event.sum() if event.sum()>0 else 0 for event in ECAL_z]

    # save features to h5 file
    oldFile.create_dataset("R9", data=np.array(R9), compression='gzip')
    oldFile.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    convertFile(inFile)
