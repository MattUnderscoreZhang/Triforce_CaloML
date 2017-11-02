# Nikolaus Howe (May 2016)
# Kaustuv Datta and Jayesh Mahaptra (July 2016)
# Maurizio Pierni (April 2017)
# Matt Zhang (May 2017)

from __future__ import division
import numpy as np
import scipy.stats
import sys
import ast
import h5py as h5
import nsub

def convertFile(inFile, outFile):

    # Open file and extract events
    oldFile = h5.File(inFile)
    newFile = h5.File(outFile, "w")

    ECAL = oldFile["ECAL"][()]*50 # Geant is in units of 1/50 GeV for some reason
    HCAL = oldFile["HCAL"][()]*50

    # if 'GAN' in inFile: # match Geant units
        # ECAL = ECAL/100
        # HCAL = HCAL/100

    # # Truth info
    # energy = oldFile["target"][()]
    # newFile.create_dataset("Energy", data=energy[:,-1]) # the way energy is saved

    # Calorimeter total energy and number of hits
    ECAL_E = np.sum(np.sum(np.sum(ECAL, axis=1), axis=1), axis=1)
    ECAL_nHits = np.sum(np.sum(np.sum(ECAL>0.1, axis=1), axis=1), axis=1)
    newFile.create_dataset("ECAL_E", data=ECAL_E)
    newFile.create_dataset("ECAL_nHits", data=ECAL_nHits)

    HCAL_E = np.sum(np.sum(np.sum(HCAL, axis=1), axis=1), axis=1)
    HCAL_nHits = np.sum(np.sum(np.sum(HCAL>0.1, axis=1), axis=1), axis=1)
    newFile.create_dataset("HCAL_E", data=HCAL_E)
    newFile.create_dataset("HCAL_nHits", data=HCAL_nHits)

    # Ratio of HCAL/ECAL energy, and other ratios
    newFile.create_dataset("HCAL_ECAL_ERatio", data=HCAL_E/ECAL_E)
    newFile.create_dataset("HCAL_ECAL_nHitsRatio", data=HCAL_nHits/ECAL_nHits)
    ECAL_E_firstLayer = np.sum(np.sum(ECAL[:,:,:,0], axis=1), axis=1)
    HCAL_E_firstLayer = np.sum(np.sum(HCAL[:,:,:,0], axis=1), axis=1)
    newFile.create_dataset("ECAL_ratioFirstLayerToTotalE", data=ECAL_E_firstLayer/ECAL_E)
    newFile.create_dataset("HCAL_ratioFirstLayerToTotalE", data=HCAL_E_firstLayer/HCAL_E)
    ECAL_E_secondLayer = np.sum(np.sum(ECAL[:,:,:,1], axis=1), axis=1)
    HCAL_E_secondLayer = np.sum(np.sum(HCAL[:,:,:,1], axis=1), axis=1)
    newFile.create_dataset("ECAL_ratioFirstLayerToSecondLayerE", data=ECAL_E_firstLayer/ECAL_E_secondLayer)
    newFile.create_dataset("HCAL_ratioFirstLayerToSecondLayerE", data=HCAL_E_firstLayer/HCAL_E_secondLayer)

    # Used in moment calculation
    # EventMinRadius = 1.5
    # SliceCellRSize = 0.00636
    # SliceCellZSize = 0.0051
    # SliceCellPhiSize = 0.0051
    EventMinRadius = 1 # chosen to make moments stay close to 1
    SliceCellRSize = 0.25
    SliceCellZSize = 0.25
    SliceCellPhiSize = 0.25

    # ECAL moments
    ECALprojX = np.sum(np.sum(ECAL, axis=3), axis=2)
    ECALprojY = np.sum(np.sum(ECAL, axis=3), axis=1)
    ECALprojZ = np.sum(np.sum(ECAL, axis=1), axis=1) # z = direction into calo
    totalE = np.sum(ECALprojX, axis=1)
    nEvents, ECAL_sizeX, ECAL_sizeY, ECAL_sizeZ = ECAL.shape 
    # ECAL_midX = (ECAL_sizeX-1)/2
    # ECAL_midY = (ECAL_sizeY-1)/2
    # ECAL_midZ = (ECAL_sizeZ-1)/2
    ECAL_midX = np.zeros(nEvents)
    ECAL_midY = np.zeros(nEvents)
    ECAL_midZ = np.zeros(nEvents)
    for i in range(6):
        print i
        relativeIndices = np.tile(np.arange(ECAL_sizeX), (nEvents,1))
        moments = np.power(abs(relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
        ECAL_momentX = np.sum(np.core.umath_tests.inner1d(ECALprojX, relativeDifferences), axis=1)/totalE
        if i==0: ECAL_midX = moments
        newFile.create_dataset("ECALmomentX" + str(i+1), data=ECAL_momentX)
    for i in range(6):
        relativeIndices = np.tile(np.arange(ECAL_sizeY), (nEvents,1))
        moments = np.power(abs(relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
        ECAL_momentY = np.sum(np.core.umath_tests.inner1d(ECALprojY, relativeDifferences), axis=1)/totalE
        if i==0: ECAL_midY = moments
        newFile.create_dataset("ECALmomentY" + str(i+1), data=ECAL_momentY)
    for i in range(6):
        relativeIndices = np.tile(np.arange(ECAL_sizeX), (nEvents,1))
        moments = np.power(abs(relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
        ECAL_momentZ = np.sum(np.core.umath_tests.inner1d(ECALprojZ, relativeDifferences), axis=1)/totalE
        if i==0: ECAL_midZ = moments
        newFile.create_dataset("ECALmomentZ" + str(i+1), data=ECAL_momentZ)

    # HCAL moments
    HCALprojX = np.sum(np.sum(HCAL, axis=3), axis=2)
    HCALprojY = np.sum(np.sum(HCAL, axis=3), axis=1)
    HCALprojZ = np.sum(np.sum(HCAL, axis=1), axis=1)
    totalE = np.sum(HCALprojX, axis=1)
    HCAL_sizeX, HCAL_sizeY, HCAL_sizeZ = HCAL[0].shape 
    # HCAL_midX = (HCAL_sizeX-1)/2
    # HCAL_midY = (HCAL_sizeY-1)/2
    # HCAL_midZ = (HCAL_sizeZ-1)/2
    HCAL_midX = np.zeros(nEvents)
    HCAL_midY = np.zeros(nEvents)
    HCAL_midZ = np.zeros(nEvents)
    for i in range(6):
        relativeIndices = np.tile(np.arange(HCAL_sizeX), (nEvents,1))
        moments = np.power(abs(relativeIndices.transpose()-HCAL_midX).transpose(), i+1)
        HCAL_momentX = np.sum(np.core.umath_tests.inner1d(HCALprojX, relativeDifferences), axis=1)/totalE
        if i==0: HCAL_midX = moments
        newFile.create_dataset("HCALmomentX" + str(i+1), data=HCAL_momentX)
    for i in range(6):
        relativeIndices = np.tile(np.arange(HCAL_sizeY), (nEvents,1))
        moments = np.power(abs(relativeIndices.transpose()-HCAL_midY).transpose(), i+1)
        HCAL_momentY = np.sum(np.core.umath_tests.inner1d(HCALprojY, relativeDifferences), axis=1)/totalE
        if i==0: HCAL_midY = moments
        newFile.create_dataset("HCALmomentY" + str(i+1), data=HCAL_momentY)
    for i in range(6):
        relativeIndices = np.tile(np.arange(HCAL_sizeX), (nEvents,1))
        moments = np.power(abs(relativeIndices.transpose()-HCAL_midZ).transpose(), i+1)
        HCAL_momentZ = np.sum(np.core.umath_tests.inner1d(HCALprojZ, relativeDifferences), axis=1)/totalE
        if i==0: HCAL_midZ = moments
        newFile.create_dataset("HCALmomentZ" + str(i+1), data=HCAL_momentZ)
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    convertFile(inFile, outFile)
