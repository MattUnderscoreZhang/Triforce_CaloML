import h5py
import numpy as np

def deleteRows(oldFile, dsetName, badRows, newFile):

    if type(badRows) is not list: badRows = [badRows] # make sure badRows is a list
    badRows.sort() # sort in increasing order

    oldDset = oldFile[dsetName]
    if oldDset.ndim == 1: # force 1D array to be 2D, with a second dimension of 1
        oldDset = np.reshape(oldDset, (-1, 1)) # numpy stacking only works with "2D arrays"

    newShape = list(oldDset.shape)
    newShape[0] -= len(badRows)
    newDset = newFile.create_dataset(dsetName, newShape) # create a new dataset of the correct shape

    #######################
    # Filling new dataset #
    #######################

    # newDsetCmd = "newDset = np.vstack(["
    # if (len(badRows) is not 1) and (badRows[0] is not 0): newDsetCmd += "oldDset[0:"+str(badRows[0])+"], "

    # for index, rowN in enumerate(badRows):

        # newDsetCmd += "oldDset["+str(rowN+1)+":"

        # if index+1 != len(badRows):
            # newDsetCmd += str(badRows[index+1])+"], "
        # else:
            # newDsetCmd += "]])"

    # exec(newDsetCmd)

    for index, rowN in enumerate(badRows):

        if (index == 0):
            if (rowN is not 0): # if the first bad index is not 0
                newDset[0:rowN] = oldDset[0:rowN] # add the slice before the first index 

        else:
            newDset[badRows[index-1]+1-index:rowN-index] = oldDset[badRows[index-1]+1:rowN] # e.g. if indices 5 and 12 are bad, add the slice [6:12]

        if (index+1 == len(badRows)):
            newDset[rowN-index:] = oldDset[rowN+1:] # fill the remainder
