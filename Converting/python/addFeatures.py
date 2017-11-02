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

    if 'GAN' in inFile: # match Geant units
        ECAL = ECAL/100
        HCAL = HCAL/100

    # Truth info
    energy = oldFile["target"][()]
    newFile.create_dataset("Energy", data=energy[:,-1]) # the way energy is saved

    # Calorimeter total energy and number of hits
    ECAL_E = np.sum(np.sum(np.sum(ECAL, axis=1), axis=1), axis=1)
    ECAL_nHits = np.sum(np.sum(np.sum(ECAL>0.1, axis=1), axis=1), axis=1)
    newFile.create_dataset("ECAL/ECAL_E", data=ECAL_E)
    newFile.create_dataset("ECAL/ECAL_nHits", data=ECAL_nHits)

    HCAL_E = np.sum(np.sum(np.sum(HCAL, axis=1), axis=1), axis=1)
    HCAL_nHits = np.sum(np.sum(np.sum(HCAL>0.1, axis=1), axis=1), axis=1)
    newFile.create_dataset("HCAL/HCAL_E", data=HCAL_E)
    newFile.create_dataset("HCAL/HCAL_nHits", data=HCAL_nHits)

    # Ratio of HCAL/ECAL energy, and other ratios
    newFile.create_dataset("HCAL_ECAL_Ratios/HCAL_ECAL_ERatio", data=HCAL_E/ECAL_E)
    newFile.create_dataset("HCAL_ECAL_Ratios/HCAL_ECAL_nHitsRatio", data=HCAL_nHits/ECAL_nHits)
    ECAL_E_firstLayer = np.sum(np.sum(ECAL[:,:,:,0], axis=1), axis=1)
    HCAL_E_firstLayer = np.sum(np.sum(HCAL[:,:,:,0], axis=1), axis=1)
    newFile.create_dataset("ECAL_Ratios/ECAL_ratioFirstLayerToTotalE", data=ECAL_E_firstLayer/ECAL_E)
    newFile.create_dataset("HCAL_Ratios/HCAL_ratioFirstLayerToTotalE", data=HCAL_E_firstLayer/HCAL_E)
    ECAL_E_secondLayer = np.sum(np.sum(ECAL[:,:,:,1], axis=1), axis=1)
    HCAL_E_secondLayer = np.sum(np.sum(HCAL[:,:,:,1], axis=1), axis=1)
    newFile.create_dataset("ECAL_Ratios/ECAL_ratioFirstLayerToSecondLayerE", data=ECAL_E_firstLayer/ECAL_E_secondLayer)
    newFile.create_dataset("HCAL_Ratios/HCAL_ratioFirstLayerToSecondLayerE", data=HCAL_E_firstLayer/HCAL_E_secondLayer)

    # Used in moment calculation
    # EventMinRadius = 1.5
    # SliceCellRSize = 0.00636
    # SliceCellZSize = 0.0051
    # SliceCellPhiSize = 0.0051
    EventMinRadius = 1 # chosen to make moments stay close to 1
    SliceCellRSize = 0.25
    SliceCellZSize = 0.25
    SliceCellPhiSize = 0.25

    # NEED TO CONVERT TO FULL-ARRAY FORM
    # ECAL moments
    ECALprojX = np.sum(np.sum(ECAL, axis=3), axis=2)
    ECALprojY = np.sum(np.sum(ECAL, axis=3), axis=1)
    ECALprojZ = np.sum(np.sum(ECAL, axis=1), axis=1) # z = direction into calo
    totalE = np.sum(ECALprojX, axis=1)
    ECAL_sizeX, ECAL_sizeY, ECAL_sizeZ = ECAL[0].shape 
    ECAL_midX = (ECAL_sizeX-1)/2
    ECAL_midY = (ECAL_sizeY-1)/2
    ECAL_midZ = (ECAL_sizeZ-1)/2
    for i in range(1, 6):
        moments = pow(abs(np.arange(ECAL_sizeX)-ECAL_midX)*EventMinRadius*SliceCellPhiSize, i+i)
        ECAL_momentX = np.sum(np.multiply(ECALprojX, moments), axis=1)/totalE
        newFile.create_dataset("ECAL_Moments/ECALmomentX" + str(i+1), data=ECAL_momentX)
    for i in range(1, 6):
        moments = pow(abs(np.arange(ECAL_sizeY)-ECAL_midY)*SliceCellZSize, i+i)
        ECAL_momentY = np.sum(np.multiply(ECALprojY, moments), axis=1)/totalE
        newFile.create_dataset("ECAL_Moments/ECALmomentY" + str(i+1), data=ECAL_momentY)
    for i in range(1, 6):
        moments = pow(abs(np.arange(ECAL_sizeZ)-ECAL_midZ)*SliceCellRSize, i+i)
        ECAL_momentZ = np.sum(np.multiply(ECALprojZ, moments), axis=1)/totalE
        newFile.create_dataset("ECAL_Moments/ECALmomentZ" + str(i+1), data=ECAL_momentZ)

    # HCAL moments
    HCALprojX = np.sum(np.sum(HCAL, axis=3), axis=2)
    HCALprojY = np.sum(np.sum(HCAL, axis=3), axis=1)
    HCALprojZ = np.sum(np.sum(HCAL, axis=1), axis=1)
    totalE = np.sum(HCALprojX, axis=1)
    HCAL_sizeX, HCAL_sizeY, HCAL_sizeZ = HCAL[0].shape 
    HCAL_midX = (HCAL_sizeX-1)/2
    HCAL_midY = (HCAL_sizeY-1)/2
    HCAL_midZ = (HCAL_sizeZ-1)/2
    for i in range(1, 6):
        moments = pow(abs(np.arange(HCAL_sizeX)-HCAL_midX)*EventMinRadius*SliceCellPhiSize, i+i)
        HCAL_momentX = np.sum(np.multiply(HCALprojX, moments), axis=1)/totalE
        newFile.create_dataset("HCAL_Moments/HCALmomentX" + str(i+1), data=HCAL_momentX)
    for i in range(1, 6):
        moments = pow(abs(np.arange(HCAL_sizeY)-HCAL_midY)*SliceCellZSize, i+i)
        HCAL_momentY = np.sum(np.multiply(HCALprojY, moments), axis=1)/totalE
        newFile.create_dataset("HCAL_Moments/HCALmomentY" + str(i+1), data=HCAL_momentY)
    for i in range(1, 6):
        moments = pow(abs(np.arange(HCAL_sizeZ)-HCAL_midZ)*SliceCellRSize, i+i)
        HCAL_momentZ = np.sum(np.multiply(HCALprojZ, moments), axis=1)/totalE
        newFile.create_dataset("HCAL_Moments/HCALmomentZ" + str(i+1), data=HCAL_momentZ)
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    convertFile(inFile, outFile)
