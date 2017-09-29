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

def convertFile(fileName):

    # Open file and extract events
    myFile = h5.File(fileName, "r+")
    ECALs = myFile["ECAL"]

    ########################
    # Calculating features #
    ########################

    myFeatures = FeaturesList()

    # Loop through all the events
    for index, ECALarray in enumerate(ECALs):

        print "Event", index, "out of", len(ECALs)

        # N-subjettiness
        eventVector = (60, 0, 0) # CHECKPOINT - change this to match event
        threshold = np.mean(ECALarray)/20 # energy threshold for a calo hit
        nsubFeatures = nsub.nsub(ECALarray, eventVector, threshold)
        myFeatures.add("N_Subjettiness/bestJets1", nsubFeatures['bestJets1'])
        myFeatures.add("N_Subjettiness/bestJets2", nsubFeatures['bestJets2'])
        myFeatures.add("N_Subjettiness/tau1", nsubFeatures['tau1'])
        myFeatures.add("N_Subjettiness/tau2", nsubFeatures['tau2'])
        myFeatures.add("N_Subjettiness/tau3", nsubFeatures['tau3'])
        myFeatures.add("N_Subjettiness/tau2_over_tau1", nsubFeatures['tau2_over_tau1'])
        myFeatures.add("N_Subjettiness/tau3_over_tau2", nsubFeatures['tau3_over_tau2'])

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
    convertFile(fileName)
