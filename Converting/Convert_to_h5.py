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

###############################################
# Jet functions for calculating nsubjettiness #
###############################################

# Find dR between two positions (eta, phi)
def dR(position1, position2):
    return np.sqrt(np.power(position1[0] - position2[0], 2) + np.power(position1[1] - position2[1], 2))

# Calculate eta (from Wikipedia)
def eta(r, z):
    p = np.sqrt(r*r + z*z)
    return 0.5 * np.log((p + z) / (p - z))

# All particles with energy above a threshold - returns [particle1, particle2...], where particle = (eta, phi, pT)
# eventVector is in the form (px, py, pz) for the calorimeter slice
def particlesAboveThreshold(caloData, threshold, eventVector):
    # CHECKPOINT - are x and y interpreted correctly?
    # (z, Rphi, r) = np.nonzero(caloData > threshold) # returns ([x], [y], [z]), which I interpret as ([z], [Rphi], [r])
    (Rphi, z, r) = np.nonzero(caloData > threshold) # returns ([x], [y], [z]), which I interpret as ([Rphi], [z], [r])
    nParticles = len(r)
    E = np.zeros(nParticles)
    for n in range(nParticles):
        # E[n] = caloData[z[n], Rphi[n], r[n]] # E of hit used as pT
        E[n] = caloData[Rphi[n], z[n], r[n]] # E of hit used as pT
    # CHECKPOINT - check these numbers
    # Detector geometry in meters, from https://twiki.cern.ch/twiki/bin/view/CLIC/ClicNDM_ECal
    EventMinRadius = 1.5
    SliceCellRSize = 0.00636
    SliceCellZSize = 0.0051
    SliceCellPhiSize = 0.0051 # in radians
    # Use event vector to determine geometric location of calorimeter slice (center of first layer)
    (px, py, pz) = eventVector
    pr = np.sqrt(px*px + py*py)
    EventPhi = np.arctan(py / px)
    EventZ = EventMinRadius * pz / pr
    # Change from indices to actual measurements in meters
    scaledZ = [(i-12)*SliceCellZSize+EventZ for i in z]
    scaledR = [i*SliceCellRSize+EventMinRadius for i in r]
    scaledPhi = [(i-12)*SliceCellPhiSize/(EventMinRadius+j)+EventPhi for i, j in zip(Rphi, scaledR)]
    scaledEta = [eta(i, j) for i, j in zip(scaledR, scaledZ)]
    return zip(scaledEta, scaledPhi, E)

# Find d_ij between two protojets
def calc_dij(v1, v2): # each vector is (eta, phi, E)
    eta1, phi1, E1 = v1
    eta2, phi2, E2 = v2
    return min(E1*E1, E2*E2) * (np.power(eta1-eta2, 2) + np.power(phi1-phi2, 2)) # letting R = 1

# based on exclusive-kT clustering algorithm from https://arxiv.org/pdf/hep-ph/9305266.pdf
# however, I force protojet combination until exactly N jets are left, ignoring the possibility of beam jets
# particles in the form [particle1, particle2...], where particle = (eta, phi, pT)
# return [[1 jet], [2 jets]... [nJets]]
# jets in form [jet1, jet2...], where jet = (eta, phi)
def antiKtJets(particles, nJets):

    setsOfBestJets = []
    nParticles = len(particles)
    protojets = particles[:]

    # If there are not enough particles, just abort early and return an empty list
    if nParticles < nJets:
        return setsOfBestJets

    # Initial values of dij
    dij = np.zeros((nParticles, nParticles))
    for n1 in range(nParticles):
        for n2 in range(n1+1, nParticles):
            dij[n1][n2] = calc_dij(particles[n1], particles[n2])
            dij[n2][n1] = dij[n1][n2] # Symmetric
    for n in range(nParticles):
        # dij[n][n] = np.power(particles[n][2], 2) # E_T squared
        dij[n][n] = 1000 # Ignore possibility of ending merge algorithm

    # If the number of particles is exactly the same as the number of requested jets, start adding to return list
    if len(protojets) <= nJets:
        bestJets = [i[0:2] for i in protojets]
        setsOfBestJets.insert(0, bestJets) # add to front of list

    # Update dij iteratively
    while(len(protojets) > 1):
        lowestIndex = np.unravel_index(dij.argmin(), dij.shape)
        if (lowestIndex[0] != lowestIndex[1]): # Merge into one jet and give it the lower of two indices
            removedProtojetIndex = max(lowestIndex[0], lowestIndex[1])
            newProtojetIndex = min(lowestIndex[0], lowestIndex[1])
            eta1, phi1, E1 = protojets[lowestIndex[0]]
            eta2, phi2, E2 = protojets[lowestIndex[1]]
            E3 = E1 + E2
            eta3 = (E1*eta1 + E2*eta2) / E3
            phi3 = (E1*phi1 + E2*phi2) / E3
            newProtojet = (eta3, phi3, E3)
            protojets.pop(removedProtojetIndex)
            dij = np.delete(dij, removedProtojetIndex, axis=0)
            dij = np.delete(dij, removedProtojetIndex, axis=1)
            protojets[newProtojetIndex] = newProtojet
            for n in range(len(protojets)):
                dij[n][newProtojetIndex] = calc_dij(protojets[n], newProtojet)
                dij[newProtojetIndex][n] = dij[n][newProtojetIndex] # Symmetric
            # dij[newProtojetIndex][newProtojetIndex] = np.power(protojets[newProtojetIndex][2], 2) # E_T squared
            dij[newProtojetIndex][newProtojetIndex] = 1000 # Ignore possibility of ending merge algorithm
        else:
            # not considered
            print "OH NO THIS SHOULDN'T BE HAPPENING!", dij[lowestIndex[0]][lowestIndex[1]]
        if len(protojets) <= nJets:
            bestJets = [i[0:2] for i in protojets]
            setsOfBestJets.insert(0, bestJets) # add to front of list

    # Return sets of best jets
    return setsOfBestJets

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

# Given an event, get the 25x25x25 array of energies around its barycentre
def getECALArray(event):
    
    # Get the event midpoint (returns absolute Y,Z weighted average at index 0,1 respectively)
    midpoint = findEventMidpoint(event)
    
    #Map absolute Y  and Z weighted average to ix and iy respectively
    #Rounding the mapping to get barycenter ix,iy values as integer (in order to select a particular cell as barycenter)
    # CHECKPOINT - this function may need updating - why is pixel [x, y] obtained by just dividing position [x, y] by 5?
    barycenter_ix = round(midpoint[0]/5)
    barycenter_iy = round(midpoint[1]/5)
    
    # Get the limit points for our grid
    # CHECKPOINT - what about wraparound?
    xmin = barycenter_ix - 12
    xmax = barycenter_ix + 12
    ymin = barycenter_iy - 12
    ymax = barycenter_iy + 12
    
    # Create the empty array to put the energies in
    # CHECKPOINT - cells are non-uniform in z after 16 layers (last layers are twice as thick, according to https://www.dropbox.com/s/ktu1ly0ge9n4jyd/CaloImagingDataset.pdf?dl=0)
    final_array = np.zeros((25, 25, 25))
    
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
    # CHECKPOINT - same issue as above
    barycenter_ix = round(midpointY/30)
    barycenter_iy = round(midpointZ/30)
    
    # Get the limit points for our grid
    # CHECKPOINT - same issue as above
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
    for index, string in enumerate(my_events_string):

        my_event = ast.literal_eval(string)

        if index%10 == 0:
            print "Event", index, "out of", len(my_events_string)

        # Make a list containing all the cell readouts of ECAL for the event and store it in a single ECAL array
        ECAL_list = []
        for cell_readout in my_event['ECAL']:
            ECAL_list.append(np.array(cell_readout))
        if len(ECAL_list)<= 0: continue

        # Get the barycenter details (The ECAL array around barycenter, Absolute Y and Z coordinates of ECAL centroid at index 0,1,2 respectively)
        ECAL_barycenter_details = getECALArray(np.array(ECAL_list))

        # Append the ECAL array of 25x25x25 cells around the barycenter to the ECAL array list
        ECALarray = ECAL_barycenter_details[0]/1000.
        myFeatures.add("ECAL/ECAL", ECALarray)

        # Make a list containing all the cell readouts of HCAL for the event and store it in a single ECAL array
        HCAL_list = []
        for cell_reading in my_event['HCAL']:
            HCAL_list.append(np.array(cell_reading))

        # Pass the absolute Y and Z cooridnates as input for determining HCAL array around barrycenter and append it to the HCAl array list
        HCALarray = getHCALArray(np.array(HCAL_list),ECAL_barycenter_details[1],ECAL_barycenter_details[2])/1000.
        myFeatures.add("HCAL/HCAL", HCALarray)

        # Calorimeter total energy and number of hits
        ECAL_E = np.sum(ECALarray)
        ECAL_hits = np.sum(ECALarray>0)
        myFeatures.add("ECAL/ECAL_E", ECAL_E)
        myFeatures.add("ECAL/ECAL_nHits", ECAL_hits)
        HCAL_E = np.sum(HCALarray)
        HCAL_hits = np.sum(HCALarray>0)
        myFeatures.add("HCAL/HCAL_E", HCAL_E)
        myFeatures.add("HCAL/HCAL_nHits", HCAL_hits)

        # Ratio of HCAL/ECAL energy, and other ratios
        myFeatures.add("HCAL_ECAL_Ratios/HCAL_ECAL_ERatio", HCAL_E/ECAL_E)
        myFeatures.add("HCAL_ECAL_Ratios/HCAL_ECAL_nHitsRatio", HCAL_hits/ECAL_hits)
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

        # Collecting particle ID, energy of hit, and 3-vector of momentum
        pdgID = my_event['pdgID']
        myFeatures.add("Event/pdgID", pdgID)
        energy = my_event['E']
        energy = energy/1000.
        myFeatures.add("Event/energy", energy)
        (px, py, pz) = (my_event['px'], my_event['py'], my_event['pz'])
        (px, py, pz) = (px/1000., py/1000., pz/1000.)
        myFeatures.add("Event/px", px)
        myFeatures.add("Event/py", py)
        myFeatures.add("Event/pz", pz)

        # N-subjettiness algorithm, as described in this paper: https://arxiv.org/pdf/1011.2268.pdf.
        eventVector = (60, 0, 0) # CHECKPOINT - change this to match event
        threshold = np.mean(ECALarray)/20 # energy threshold for a calo hit
        particles = particlesAboveThreshold(ECALarray, threshold, eventVector) # particles in the form (eta, phi, pT)
        # After identifying N candidate subjets, we calculate tauN.
        # Ignoring characteristic jet radius, since I'm focused on exclusive jet clustering, and it's only a scalar for taus.
        tauN = []
        # Return the sets of best 1, 2, and 3 jets
        # Jets reconstructed using anti-kT algorithm here: https://arxiv.org/pdf/hep-ph/9305266.pdf (use only ECAL data).
        nJets = 3
        jetSets = antiKtJets(particles, nJets) 
        if len(jetSets) > 0: # if jet-finding succeeded
            for jets in jetSets:
                tau = 0
                d0 = 0 # normalization factor
                for particle in particles:
                    dRList = []
                    particlePosition = (particle[0], particle[1])
                    for jet in jets:
                        dRList.append(dR(particlePosition, jet))
                    tau += particle[2] * min(dRList) # pT * min(dR)
                    d0 += particle[2] # pT
                tau = tau/d0
                tauN.append(tau) 
            myFeatures.add("N_Subjettiness/bestJets1", jetSets[1][0]) # 1st of 2 best jets
            myFeatures.add("N_Subjettiness/bestJets2", jetSets[1][1]) # 2nd of 2 best jets
        else:
            for n in range(nJets):
                tauN.append(1000) # put in a ridiculously large tau value 
            myFeatures.add("N_Subjettiness/bestJets1", [0, 0])
            myFeatures.add("N_Subjettiness/bestJets2", [0, 0])
        myFeatures.add("N_Subjettiness/tau1", tauN[0])
        myFeatures.add("N_Subjettiness/tau2", tauN[1])
        myFeatures.add("N_Subjettiness/tau3", tauN[2])
        myFeatures.add("N_Subjettiness/tau2_over_tau1", tauN[1]/tauN[0])
        myFeatures.add("N_Subjettiness/tau3_over_tau2", tauN[2]/tauN[1])

        # Opening angle
        openingAngle = my_event['openingAngle']
        myFeatures.add("OpeningAngle", openingAngle)

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
