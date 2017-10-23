from __future__ import division
import numpy as np

##############################################################################################
# N-subjettiness algorithm, as described in this paper: https://arxiv.org/pdf/1011.2268.pdf. #
##############################################################################################

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
    # Assuming the ECAL has been downsampled from an initial slice of size 25x25x25, we determine the sizes of the new cells
    sizeX, sizeY, sizeZ = caloData.shape
    SliceCellRSize *= (25.0/sizeZ)
    SliceCellZSize *= (25.0/sizeY)
    SliceCellPhiSize *= (25.0/sizeX)
    centerZ = (sizeY-1)/2.0
    centerPhi = (sizeX-1)/2.0
    # Use event vector to determine geometric location of calorimeter slice (center of first layer)
    (px, py, pz) = eventVector
    pr = np.sqrt(px*px + py*py)
    EventPhi = np.arctan(py / px)
    EventZ = EventMinRadius * pz / pr
    # Change from indices to actual measurements in meters
    scaledZ = [(i-centerZ)*SliceCellZSize+EventZ for i in z]
    scaledR = [i*SliceCellRSize+EventMinRadius for i in r]
    scaledPhi = [(i-centerPhi)*SliceCellPhiSize/(EventMinRadius+j)+EventPhi for i, j in zip(Rphi, scaledR)]
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

#########################
# Main calling function #
#########################

def nsub(ECALarray, eventVector, threshold = 0):
    particles = particlesAboveThreshold(ECALarray, threshold, eventVector) # particles in the form (eta, phi, pT)
    # After identifying N candidate subjets, we calculate tauN.
    # Ignoring characteristic jet radius, since I'm focused on exclusive jet clustering, and it's only a scalar for taus.
    tauN = []
    # Return the sets of best 1, 2, and 3 jets
    # Jets reconstructed using anti-kT algorithm here: https://arxiv.org/pdf/hep-ph/9305266.pdf (use only ECAL data).
    nJets = 3
    jetSets = antiKtJets(particles, nJets) 
    myFeatures = {}
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
        myFeatures["bestJets1"] = jetSets[1][0] # 1st of 2 best jets
        myFeatures["bestJets2"] = jetSets[1][1] # 2nd of 2 best jets
    else:
        for n in range(nJets):
            tauN.append(1000) # put in a ridiculously large tau value 
        myFeatures["bestJets1"] = [0, 0]
        myFeatures["bestJets2"] = [0, 0]
    myFeatures["tau1"] = tauN[0]
    myFeatures["tau2"] = tauN[1]
    myFeatures["tau3"] = tauN[2]
    myFeatures["tau2_over_tau1"] = tauN[1]/tauN[0]
    myFeatures["tau3_over_tau2"] = tauN[2]/tauN[1]
    return myFeatures
