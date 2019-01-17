from __future__ import division
import numpy as np
import scipy.stats
import sys
import ast
import h5py as h5
import nsub
from featuresList import FeaturesList

############################
# File reading and writing #
############################

def fixFile(fileName):

    # Open file and extract events
    myFile = h5.File(fileName, "r+")
    ECALs = myFile["ECAL"]
    HCALs = myFile["HCAL"]

    # Delete existing features
    badFeatures = ['ECAL_ratioFirstLayerToSecondLayerE', 'ECAL_ratioFirstLayerToTotalE', 'ECALmomentX0', 'ECALmomentX1', 'ECALmomentX2', 'ECALmomentX3', 'ECALmomentX4', 'ECALmomentX5', 'ECALmomentY0', 'ECALmomentY1', 'ECALmomentY2', 'ECALmomentY3', 'ECALmomentY4', 'ECALmomentY5', 'ECALmomentZ0', 'ECALmomentZ1', 'ECALmomentZ2', 'ECALmomentZ3', 'ECALmomentZ4', 'ECALmomentZ5', 'HCAL_ratioFirstLayerToSecondLayerE', 'HCAL_ratioFirstLayerToTotalE', 'HCALmomentX0', 'HCALmomentX1', 'HCALmomentX2', 'HCALmomentX3', 'HCALmomentX4', 'HCALmomentX5', 'HCALmomentY0', 'HCALmomentY1', 'HCALmomentY2', 'HCALmomentY3', 'HCALmomentY4', 'HCALmomentY5', 'HCALmomentZ0', 'HCALmomentZ1', 'HCALmomentZ2', 'HCALmomentZ3', 'HCALmomentZ4', 'HCALmomentZ5']
    for feature in badFeatures:
        del myFile[feature]

    ########################
    # Calculating features #
    ########################

    myFeatures = FeaturesList()

    # Loop through all the events
    for index, CALs in enumerate(zip(ECALs, HCALs)):

        print "Event", index, "out of", len(ECALs)

        ECALarray, HCALarray = CALs

        # Calorimeter total energy and number of hits
        ECAL_E = np.sum(ECALarray)
        ECAL_hits = np.sum(ECALarray>0)
        HCAL_E = np.sum(HCALarray)
        HCAL_hits = np.sum(HCALarray>0)

        # Ratio of HCAL/ECAL energy, and other ratios
        ECAL_E_firstLayer = np.sum(ECALarray[:,:,0]) # [x, y, z], where z is layer number perpendicular to incidence w/ calo
        HCAL_E_firstLayer = np.sum(HCALarray[:,:,0])
        myFeatures.add("ECAL_Ratios/ECAL_ratioFirstLayerToTotalE", ECAL_E_firstLayer/ECAL_E)
        myFeatures.add("HCAL_Ratios/HCAL_ratioFirstLayerToTotalE", HCAL_E_firstLayer/HCAL_E)
        ECAL_E_secondLayer = np.sum(ECALarray[:,:,1])
        HCAL_E_secondLayer = np.sum(HCALarray[:,:,1])
        myFeatures.add("ECAL_Ratios/ECAL_ratioFirstLayerToSecondLayerE", ECAL_E_firstLayer/ECAL_E_secondLayer)
        myFeatures.add("HCAL_Ratios/HCAL_ratioFirstLayerToSecondLayerE", HCAL_E_firstLayer/HCAL_E_secondLayer)

        # ECAL moments
        ECALprojX = np.sum(np.sum(ECALarray, axis=2), axis=1)
        ECALprojY = np.sum(np.sum(ECALarray, axis=2), axis=0)
        ECALprojZ = np.sum(np.sum(ECALarray, axis=0), axis=0) # z = direction into calo
        for i in range(6):
            ECAL_momentX = scipy.stats.moment(ECALprojX, moment=i+1)
            myFeatures.add("ECAL_Moments/ECALmomentX" + str(i+1), ECAL_momentX)
        for i in range(6):
            ECAL_momentY = scipy.stats.moment(ECALprojY, moment=i+1)
            myFeatures.add("ECAL_Moments/ECALmomentY" + str(i+1), ECAL_momentY)
        for i in range(6):
            ECAL_momentZ = scipy.stats.moment(ECALprojZ, moment=i+1)
            myFeatures.add("ECAL_Moments/ECALmomentZ" + str(i+1), ECAL_momentZ)

        # HCAL moments
        HCALprojX = np.sum(np.sum(HCALarray, axis=2), axis=1)
        HCALprojY = np.sum(np.sum(HCALarray, axis=2), axis=0)
        HCALprojZ = np.sum(np.sum(HCALarray, axis=0), axis=0)
        for i in range(6):
            HCAL_momentX = scipy.stats.moment(HCALprojX, moment=i+1)
            myFeatures.add("HCAL_Moments/HCALmomentX" + str(i+1), HCAL_momentX)
        for i in range(6):
            HCAL_momentY = scipy.stats.moment(HCALprojY, moment=i+1)
            myFeatures.add("HCAL_Moments/HCALmomentY" + str(i+1), HCAL_momentY)
        for i in range(6):
            HCAL_momentZ = scipy.stats.moment(HCALprojZ, moment=i+1)
            myFeatures.add("HCAL_Moments/HCALmomentZ" + str(i+1), HCAL_momentZ)

    # Save features to h5 file
    for key in myFeatures.keys():
        myFile.create_dataset(key, data=np.array(myFeatures.get(key)).squeeze(),compression='gzip')
    myFile.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    fileName = sys.argv[1]
    fixFile(fileName)
