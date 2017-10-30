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

    ECAL = oldFile["ECAL"][()]
    HCAL = oldFile["HCAL"][()]
    newFile.create_dataset("ECAL/ECAL", data=ECAL)
    newFile.create_dataset("HCAL/HCAL", data=HCAL)

    # Calorimeter total energy and number of hits
    ECAL_E = np.sum(np.sum(np.sum(ECAL, axis=1), axis=1), axis=1)
    ECAL_nHits = np.sum(np.sum(np.sum(ECAL>0, axis=1), axis=1), axis=1)
    newFile.create_dataset("ECAL/ECAL_E", data=ECAL_E)
    newFile.create_dataset("ECAL/ECAL_nHits", data=ECAL_nHits)

    HCAL_E = np.sum(np.sum(np.sum(HCAL, axis=1), axis=1), axis=1)
    HCAL_nHits = np.sum(np.sum(np.sum(HCAL>0, axis=1), axis=1), axis=1)
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

    # ECAL moments
    ECALprojX = np.sum(np.sum(ECAL, axis=3), axis=2)
    ECALprojY = np.sum(np.sum(ECAL, axis=3), axis=1)
    ECALprojZ = np.sum(np.sum(ECAL, axis=1), axis=1) # z = direction into calo
    for i in range(1, 6):
        ECAL_momentX = np.array([scipy.stats.moment(j, moment=i+1) for j in ECALprojX])
        newFile.create_dataset("ECAL_Moments/ECALmomentX" + str(i+1), data=ECAL_momentX)
    for i in range(1, 6):
        ECAL_momentY = np.array([scipy.stats.moment(j, moment=i+1) for j in ECALprojY])
        newFile.create_dataset("ECAL_Moments/ECALmomentY" + str(i+1), data=ECAL_momentY)
    for i in range(1, 6):
        ECAL_momentZ = np.array([scipy.stats.moment(j, moment=i+1) for j in ECALprojZ])
        newFile.create_dataset("ECAL_Moments/ECALmomentZ" + str(i+1), data=ECAL_momentZ)

    # HCAL moments
    HCALprojX = np.sum(np.sum(HCAL, axis=3), axis=2)
    HCALprojY = np.sum(np.sum(HCAL, axis=3), axis=1)
    HCALprojZ = np.sum(np.sum(HCAL, axis=1), axis=1)
    for i in range(1, 6):
        HCAL_momentX = np.array([scipy.stats.moment(j, moment=i+1) for j in HCALprojX])
        newFile.create_dataset("HCAL_Moments/HCALmomentX" + str(i+1), data=HCAL_momentX)
    for i in range(1, 6):
        HCAL_momentY = np.array([scipy.stats.moment(j, moment=i+1) for j in HCALprojY])
        newFile.create_dataset("HCAL_Moments/HCALmomentY" + str(i+1), data=HCAL_momentY)
    for i in range(1, 6):
        HCAL_momentZ = np.array([scipy.stats.moment(j, moment=i+1) for j in HCALprojZ])
        newFile.create_dataset("HCAL_Moments/HCALmomentZ" + str(i+1), data=HCAL_momentZ)
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    convertFile(inFile, outFile)
