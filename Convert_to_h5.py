# Nikolaus Howe (May 2016)
# Kaustuv Datta and Jayesh Mahaptra (July 2016)
# Maurizio Pierni (April 2017)
# Matt Zhang (May 2017)

import numpy as np
import scipy.stats
import sys
import ast
import h5py

##################################
# Functions for finding centroid #
##################################

# Given a list of distances and corresponding weights, calculate the weighted average
def findMidpoint(distance, energy):
    return np.average(distance, weights = energy)
    
# Given an array of interactions (ix,iy,iz,E,X,Y,Z), returns the weighted average (aveY, aveZ)
def findEventMidpoint(event):
    Yave = findMidpoint(event[:,5], event[:,3]) 
    Zave = findMidpoint(event[:,6], event[:,3]) 
    return (Yave, Zave)

##################################################
# Check if between bounds for calorimeter window #
##################################################

#Checking range for ECAL (including min and max as the total number of cells is odd)
def withinEcal(value, mymin, mymax):
    return (value >= mymin and value <= mymax)

#Checking range for HCAL (including min and max as the total number of cells is odd)     
def withinHcal(value, mymin, mymax):
    return (value >= mymin and value <= mymax)

########################################
# Implementation of calorimeter window #
########################################

# Given an event, get the 20x20x25 array of energies around its barycentre
def getECALArray(event):
    
    # Get the event midpoint (returns absolute Y,Z weighted average at index 0,1 respectively)
    midpoint = findEventMidpoint(event)
    
    #Map absolute Y  and Z weighted average to ix and iy respectively
    #Rounding the mapping to get barycenter ix,iy values as integer (in order to select a particular cell as barycenter)
    barycenter_ix = round(midpoint[0]/5)
    barycenter_iy = round(midpoint[1]/5)
    
    # Get the limit points for our grid
    xmin = barycenter_ix - 12
    xmax = barycenter_ix + 12
    ymin = barycenter_iy - 12
    ymax = barycenter_iy + 12
    
    # Create the empty array to put the energies in
    final_array = np.zeros((25, 25, 25))
    
    # Fill the array with energy values, if they exist
    #for element in event:
    #    if within(element[0], xmin, xmax) and within(element[1], ymin, ymax) and within(element[2], zmin, zmax):
    #        final_array[element[0], element[1], element[2]] = element[3]
  
    # Fill the array with energy values, if they exist
    for ix, iy, iz, E, x, y, z in event:
        if withinEcal(ix, xmin, xmax) and withinEcal(iy, ymin, ymax):
            final_array[ix-xmin,iy-ymin,iz] = E
    return final_array,midpoint[0],midpoint[1]

# Given an event and Absolute Y and Z coordinates of the ECAL centroid
# get the 4x4x60 array of energies around the same coordinates of HCAL
def getHCALArray(event,midpointY,midpointZ):
    
    # Use the Y and Z of ECAL centroid
   
    #Map absolute Y and Z weighted average to ix and iy respectively
    #Rounding the mapping to get barycenter ix,iy values as integer (in order to select a particular cell as barycenter)
    barycenter_ix = round(midpointY/30)
    barycenter_iy = round(midpointZ/30)
    
    # Get the limit points for our grid
    xmin = barycenter_ix - 2
    xmax = barycenter_ix + 2
    ymin = barycenter_iy - 2
    ymax = barycenter_iy + 2
    
    # Create the empty array to put the energies in
    final_array = np.zeros((5, 5, 60))
    
    # Fill the array with energy values, if they exist
    #for element in event:
    #    if within(element[0], xmin, xmax) and within(element[1], ymin, ymax) and within(element[2], zmin, zmax):
    #        final_array[element[0], element[1], element[2]] = element[3]
  
    # Fill the array with energy values, if they exist
    for ix, iy, iz, E, x, y, z in event:
        if withinHcal(ix, xmin, xmax) and withinHcal(iy, ymin, ymax):
            final_array[ix-xmin,iy-ymin,iz] = E
    return final_array

############
# Features #
############

class FeaturesList(object):

    def __init__(self):
        self.features = {}

    def add(self, featureName, feature):
        self.features.setdefault(featureName, []).append(feature)

    def keys(self):
        return self.features.keys()

    def get(self, featureName):
        return self.features[featureName]

############################
# File reading and writing #
############################

def convertFile(inFile, outFile):

    # Open file and extract events
    with open(inFile) as myfile:
        my_events_string = myfile.read().replace('\n', '')
    my_events_string = my_events_string.replace(' ', '')
    my_events_string = my_events_string.replace('}{','} {')
    my_events_string = my_events_string.split()

    ########################
    # Calculating features #
    ########################

    myFeatures = FeaturesList()

    # Loop through all the events
    for string in my_events_string:

        my_event = ast.literal_eval(string)

        # Make a list containing all the cell readouts of ECAL for the event and store it in a single ECAL array
        ECAL_list = []
        for cell_readout in my_event['ECAL']:
            ECAL_list.append(np.array(cell_readout))
        if len(ECAL_list)<= 0: continue

        # Get the barycenter details (The ECAL array around barycenter, Absolute Y and Z coordinates of ECAL centroid at index 0,1,2 respectively)
        ECAL_barycenter_details = getECALArray(np.array(ECAL_list))

        # Append the ECAL array of 25x25x25 cells around the barycenter to the ECAL array list
        ECALarray = ECAL_barycenter_details[0]/1000.
        myFeatures.add("ECAL", ECALarray)

        # Make a list containing all the cell readouts of HCAL for the event and store it in a single ECAL array
        HCAL_list = []
        for cell_reading in my_event['HCAL']:
            HCAL_list.append(np.array(cell_reading))

        # Pass the absolute Y and Z cooridnates as input for determining HCAL array around barrycenter and append it to the HCAl array list
        HCALarray = getHCALArray(np.array(HCAL_list),ECAL_barycenter_details[1],ECAL_barycenter_details[2])/1000.
        myFeatures.add("HCAL", HCALarray)

        # Calorimeter total energy and number of hits
        ECAL_E = np.sum(ECALarray)
        ECAL_hits = np.sum(ECALarray>0)
        myFeatures.add("ECAL_E", ECAL_E)
        myFeatures.add("ECAL_nHits", ECAL_hits)
        HCAL_E = np.sum(HCALarray)
        HCAL_hits = np.sum(HCALarray>0)
        myFeatures.add("HCAL_E", HCAL_E)
        myFeatures.add("HCAL_nHits", HCAL_hits)

        # Ratio of ECAL/HCAL energy, and other ratios
        myFeatures.add("ECAL_HCAL_ERatio", ECAL_E/HCAL_E)
        myFeatures.add("ECAL_HCAL_nHitsRatio", ECAL_hits/HCAL_hits)
        ECAL_E_firstLayer = np.sum(ECALarray[0])
        HCAL_E_firstLayer = np.sum(HCALarray[0])
        myFeatures.add("ECAL_ratioFirstLayerToTotalE", ECAL_E_firstLayer/ECAL_E)
        myFeatures.add("HCAL_ratioFirstLayerToTotalE", HCAL_E_firstLayer/HCAL_E)
        ECAL_E_secondLayer = np.sum(ECALarray[1])
        HCAL_E_secondLayer = np.sum(HCALarray[1])
        myFeatures.add("ECAL_ratioFirstLayerToSecondLayerE", ECAL_E_firstLayer/ECAL_E_secondLayer)
        myFeatures.add("HCAL_ratioFirstLayerToSecondLayerE", HCAL_E_firstLayer/HCAL_E_secondLayer)

        # ECAL moments
        ECALprojX = np.sum(np.sum(ECALarray, axis=2), axis=1)
        ECALprojY = np.sum(np.sum(ECALarray, axis=2), axis=0)
        ECALprojZ = np.sum(np.sum(ECALarray, axis=0), axis=0)
        for i in range(6):
            ECAL_momentX = scipy.stats.moment(ECALprojX, moment=i+1)
            myFeatures.add("ECALmomentX" + str(i), ECAL_momentX)
        for i in range(6):
            ECAL_momentY = scipy.stats.moment(ECALprojY, moment=i+1)
            myFeatures.add("ECALmomentY" + str(i), ECAL_momentY)
        for i in range(6):
            ECAL_momentZ = scipy.stats.moment(ECALprojZ, moment=i+1)
            myFeatures.add("ECALmomentZ" + str(i), ECAL_momentZ)

        # HCAL moments
        HCALprojX = np.sum(np.sum(HCALarray, axis=2), axis=1)
        HCALprojY = np.sum(np.sum(HCALarray, axis=2), axis=0)
        HCALprojZ = np.sum(np.sum(HCALarray, axis=0), axis=0)
        for i in range(6):
            HCAL_momentX = scipy.stats.moment(HCALprojX, moment=i+1)
            myFeatures.add("HCALmomentX" + str(i), HCAL_momentX)
        for i in range(6):
            HCAL_momentY = scipy.stats.moment(HCALprojY, moment=i+1)
            myFeatures.add("HCALmomentY" + str(i), HCAL_momentY)
        for i in range(6):
            HCAL_momentZ = scipy.stats.moment(HCALprojZ, moment=i+1)
            myFeatures.add("HCALmomentZ" + str(i), HCAL_momentZ)

        # Collecting particle ID, energy of hit, and 3-vector of momentum
        pdgID = my_event['pdgID']
        #if pdgID == 211 or pdgID == 111:
        #    pdgID = 0
        #if pdgID == 22:
        #    pdgID = 1
        energy = my_event['E']
        energy = energy/1000.
        #(px, py, pz) = (my_event['px'], my_event['py'], my_event['pz'])
        #(px, py, pz) = (px/1000., py/1000., pz/1000.)
        #target = np.zeros((1, 5))
        #target[:,0], target[:,1], target[:,2], target[:,3], target[:,4] = (pdgID, energy, px, py, pz)
        # target = np.zeros(2)
        # target[0], target[1] = (pdgID, energy)
        # target_array_list.append(target)
        myFeatures.add("pdgID", pdgID)
        myFeatures.add("energy", energy)

    # Save features to an h5 file
    f = h5py.File(outFile, "w")
    for key in myFeatures.keys():
        f.create_dataset(key, data=np.array(myFeatures.get(key)).squeeze(),compression='gzip')
    f.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    convertFile(inFile, outFile)
