import h5py
import numpy as np

def deleteRows(oldFile, dsetName, badRows, newFile):

    if type(badRows) is not list: badRows = [badRows] # make sure badRows is a list
    badRows.sort() # sort in increasing order

    oldDset = oldFile[dsetName]
    if len(oldDset.shape) == 1: # force 1D array to be 2D, with a second dimension of 1
        oldDset = np.reshape(oldDset, (-1, 1)) # numpy stacking only works with "2D arrays"

    newShape = list(oldDset.shape)
    newShape[0] -= len(badRows)
    newDset = newFile.create_dataset(dsetName, newShape) # create a new dataset of the correct shape

    #######################
    # Filling new dataset #
    #######################

    for index, rowN in enumerate(badRows):

        if (index == 0):
            if (rowN is not 0): # if the first bad index is not 0
                newDset[0:rowN] = oldDset[0:rowN] # add the slice before the first index 

        elif (badRows[index-1]+1 != rowN): # ignore zero-length slices
            newDset[badRows[index-1]+1-index:rowN-index] = oldDset[badRows[index-1]+1:rowN] # e.g. if indices 5 and 12 are bad, add the slice [6:12]

        if (index+1 == len(badRows) and rowN+1 != len(oldDset)):
            newDset[rowN-index:] = oldDset[rowN+1:] # fill the remainder
