# Nikolaus Howe (May 2016)
# Kaustuv Datta and Jayesh Mahaptra (July 2016)
# Maurizio Pierni (April 2017)
# Matt Zhang (May 2017)

from __future__ import division
import numpy as np
import scipy.stats
import sys
import ast
import h5py
import nsub
from featuresList import FeaturesList

import os

import collections
import glob

############################
# File reading and writing #
############################

def convertFile(inFile, outFile, pdgID):

    ########################
    # Calculating features #
    ########################

    files=sorted(glob.glob(inFile))

    myFeatures = collections.OrderedDict()

    # Loop through all the events
    np.seterr(divide='ignore')

    for file_index in range(len(files)):

        f1 = h5py.File(files[file_index], 'r')

        ECALarray=f1["ECAL"][:]
        ECALarray=ECALarray.astype(np.float32)

        HCALarray=f1["HCAL"][:]
        HCALarray=HCALarray.astype(np.float32)

        myFeatures["ECAL"]= ECALarray

        myFeatures["HCAL"]= HCALarray

        # Calorimeter total energy and number of hits
        ECAL_E = np.sum(ECALarray, axis=(1,2,3))
        ECAL_hits = np.sum(ECALarray>0, axis=(1,2,3))
        myFeatures["ECAL_E"]= ECAL_E
        myFeatures["ECAL_nHits"]= ECAL_hits
        HCAL_E = np.sum(HCALarray, axis=(1,2,3))
        HCAL_hits = np.sum(HCALarray>0, axis=(1,2,3))
        myFeatures["HCAL_E"]= HCAL_E
        myFeatures["HCAL_nHits"]= HCAL_hits

        # Ratio of HCAL/ECAL energy, and other ratios
        myFeatures["HCAL_ECAL_ERatio"]= HCAL_E/ECAL_E
        myFeatures["HCAL_ECAL_nHitsRatio"]= HCAL_hits/ECAL_hits
        
        #ECAL_E_firstLayer = np.sum(ECALarray[:,:,0]) # [x, y, z], where z is layer number perpendicular to incidence w/ calo
        #HCAL_E_firstLayer = np.sum(HCALarray[:,:,0])
        ECAL_E_firstLayer = np.sum(ECALarray, axis=(1,2))[:,0]
        HCAL_E_firstLayer = np.sum(HCALarray, axis=(1,2))[:,0]
        myFeatures["ECAL_ratioFirstLayerToTotalE"]= ECAL_E_firstLayer/ECAL_E
        myFeatures["HCAL_ratioFirstLayerToTotalE"]= HCAL_E_firstLayer/HCAL_E
        #ECAL_E_secondLayer = np.sum(ECALarray[:,:,1])
        #HCAL_E_secondLayer = np.sum(HCALarray[:,:,1])
        ECAL_E_secondLayer = np.sum(ECALarray, axis=(1,2))[:,1]
        HCAL_E_secondLayer = np.sum(HCALarray, axis=(1,2))[:,1]
        myFeatures["ECAL_ratioFirstLayerToSecondLayerE"]= ECAL_E_firstLayer/ECAL_E_secondLayer
        myFeatures["HCAL_ratioFirstLayerToSecondLayerE"]= HCAL_E_firstLayer/HCAL_E_secondLayer

        # ECAL moments
        #ECALprojX = np.sum(np.sum(ECALarray, axis=2), axis=1)
        #ECALprojY = np.sum(np.sum(ECALarray, axis=2), axis=0)
        #ECALprojZ = np.sum(np.sum(ECALarray, axis=0), axis=0) # z = direction into calo
        ECALprojX = np.sum(np.sum(ECALarray, axis=3), axis=2)
        ECALprojY = np.sum(np.sum(ECALarray, axis=3), axis=1)
        ECALprojZ = np.sum(np.sum(ECALarray, axis=1), axis=1)
        for i in range(1,6):
            ECAL_momentX = scipy.stats.moment(ECALprojX, moment=i+1, axis=1)
            myFeatures["ECALmomentX" + str(i+1)]=ECAL_momentX
        for i in range(1,6):
            ECAL_momentY = scipy.stats.moment(ECALprojY, moment=i+1, axis=1)
            myFeatures["ECALmomentY" + str(i+1)]=ECAL_momentY
        for i in range(1,6):
            ECAL_momentZ = scipy.stats.moment(ECALprojZ, moment=i+1, axis=1)
            myFeatures["ECALmomentZ" + str(i+1)]=ECAL_momentZ
        #for i in range(6):
            #ECAL_momentX = scipy.stats.moment(ECALprojX, moment=i+1)
            #myFeatures["ECALmomentX" + str(i+1)]= ECAL_momentX
        #for i in range(6):
            #ECAL_momentY = scipy.stats.moment(ECALprojY, moment=i+1)
            #myFeatures["ECALmomentY" + str(i+1)]= ECAL_momentY
        #for i in range(6):
            #ECAL_momentZ = scipy.stats.moment(ECALprojZ, moment=i+1)
            #myFeatures["ECALmomentZ" + str(i+1)]=ECAL_momentZ

        # HCAL moments
        HCALprojX = np.sum(np.sum(HCALarray, axis=3), axis=2)
        HCALprojY = np.sum(np.sum(HCALarray, axis=3), axis=1)
        HCALprojZ = np.sum(np.sum(HCALarray, axis=1), axis=1)
        for i in range(1,6):
            HCAL_momentX = scipy.stats.moment(HCALprojX, moment=i+1, axis=1)
            myFeatures["HCALmomentX" + str(i+1)]=HCAL_momentX
        for i in range(1,6):
            HCAL_momentY = scipy.stats.moment(HCALprojY, moment=i+1, axis=1)
            myFeatures["HCALmomentY" + str(i+1)]=HCAL_momentY
        for i in range(1,6):
            HCAL_momentZ = scipy.stats.moment(HCALprojZ, moment=i+1, axis=1)
            myFeatures["HCALmomentZ" + str(i+1)]=HCAL_momentZ
        #HCALprojX = np.sum(np.sum(HCALarray, axis=2), axis=1)
        #HCALprojY = np.sum(np.sum(HCALarray, axis=2), axis=0)
        #HCALprojZ = np.sum(np.sum(HCALarray, axis=0), axis=0)
        #for i in range(1,6):
            #HCAL_momentX = scipy.stats.moment(HCALprojX, moment=i+1)
            #myFeatures["HCALmomentX" + str(i+1)]=HCAL_momentX
        #for i in range(6):
            #HCAL_momentY = scipy.stats.moment(HCALprojY, moment=i+1)
            #myFeatures["HCALmomentY" + str(i+1)]=HCAL_momentY
        #for i in range(6):
            #HCAL_momentZ = scipy.stats.moment(HCALprojZ, moment=i+1)
            #myFeatures["HCALmomentZ" + str(i+1)]=HCAL_momentZ

        # Collecting particle ID, energy of hit, and 3-vector of momentum
        #pdgID = my_event['pdgID']
        myFeatures["pdgID"]= np.full((10000,1), pdgID)
        #myFeatures["pdgID"]= pdgID
        #energy = my_event['E']
        #energy = energy/1000.
        #myFeatures["energy"]= energy
        #(px, py, pz) = (my_event['px'], my_event['py'], my_event['pz'])
        #(px, py, pz) = (px/1000., py/1000., pz/1000.)
        #myFeatures["px"]= px
        #myFeatures["py"]= py
        #myFeatures["pz"]= pz

        # N-subjettiness
        eventVector = (60, 0, 0) # CHECKPOINT - change this to match event
        threshold = np.mean(ECALarray, axis=(1,2,3))/20 # energy threshold for a calo hit
        assert(ECALarray.shape[0]==10000)

        bJ10=[]
        bJ11=[]
        bJ20=[]
        bJ21=[]
        t1=[]
        t2=[]
        t3=[]
        t21=[]
        t32=[]

        for i in range(10000):
            nsubFeatures = nsub.nsub(ECALarray[i][:], eventVector, threshold[i])
            bJ10.append(nsubFeatures['bestJets1'][0])
            bJ11.append(nsubFeatures['bestJets1'][1])
            bJ20.append(nsubFeatures['bestJets2'][0])
            bJ21.append(nsubFeatures['bestJets2'][1])
            t1.append(nsubFeatures['tau1'])
            t2.append(nsubFeatures['tau2'])
            t3.append(nsubFeatures['tau3'])

            t21.append(nsubFeatures['tau2_over_tau1'])
            t32.append(nsubFeatures['tau3_over_tau2'])

        myFeatures["bestJets10"]= np.array(bJ10)
        myFeatures["bestJets11"]= np.array(bJ11)
        myFeatures["bestJets20"]= np.array(bJ20)
        myFeatures["bestJets21"]= np.array(bJ21)
        myFeatures["tau1"]= np.array(t1)
        myFeatures["tau2"]=np.array(t2)
        myFeatures["tau3"]=np.array(t3)
        myFeatures["tau2_over_tau1"]=np.array(t21)
        myFeatures["tau3_over_tau2"]=np.array(t32)

        # Opening angle
        #openingAngle = my_event['openingAngle']
        #myFeatures["OpeningAngle"]= openingAngle

        # Save features to an h5 file
        f = h5py.File(outFile+os.path.basename(files[file_index]), "w")
        for ['ECAL', 'HCAL']:
            f.create_dataset(key, data=myFeatures[key])
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    pdgID = float(sys.argv[3])
    convertFile(inFile, outFile, pdgID)
