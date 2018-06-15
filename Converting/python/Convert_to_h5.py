# Nikolaus Howe (May 2016)
# Kaustuv Datta and Jayesh Mahaptra (July 2016)
# Maurizio Pierni (April 2017)
# Matt Zhang (May 2017)
# Dominick Olivito (Jan 2018)

from __future__ import division
import numpy as np
import scipy.stats
import sys
import ast
import h5py
import nsub
import os
import math
from featuresList import FeaturesList
import addFeatures

#####################################
# Maximum abs(ix) for each iz layer #
#####################################

## should be complete
max_ix_perlayer_ECAL = {
    0 : 79,
    1 : 79,
    2 : 79,
    3 : 80,
    4 : 80,
    5 : 80,
    6 : 80,
    7 : 80,
    8 : 81,
    9 : 81,
    10 : 81,
    11 : 81,
    12 : 82,
    13 : 82,
    14 : 82,
    15 : 82,
    16 : 83,
    17 : 83,
    18 : 83,
    19 : 83,
    20 : 84,
    21 : 84,
    22 : 85,
    23 : 85,
    24 : 85,
}

## should be complete, based on charged pion samples
max_ix_perlayer_HCAL = {
    0 : 15,
    1 : 15,
    2 : 16,
    3 : 16,
    4 : 16,
    5 : 16,
    6 : 17,
    7 : 17,
    8 : 17,
    9 : 17,
    10 : 18,
    11 : 18,
    12 : 18,
    13 : 18,
    14 : 18,
    15 : 19,
    16 : 19,
    17 : 19,
    18 : 19,
    19 : 20,
    20 : 20,
    21 : 20,
    22 : 20,
    23 : 21,
    24 : 21,
    25 : 21,
    26 : 21,
    27 : 22,
    28 : 22,
    29 : 22,
    30 : 22,
    31 : 22,
    32 : 23,
    33 : 23,
    34 : 23,
    35 : 23,
    36 : 24,
    37 : 24,
    38 : 24,
    39 : 24,
    40 : 25,
    41 : 25,
    42 : 25,
    43 : 25,
    44 : 26,
    45 : 26,
    46 : 26,
    47 : 26,
    48 : 27,
    49 : 27,
    50 : 27,
    51 : 27,
    52 : 27,
    53 : 28,
    54 : 28,
    55 : 28,
    56 : 28,
    57 : 29,
    58 : 29,
    59 : 29,
}

# returns new array using dictionaries above
# based on: https://stackoverflow.com/a/16992881
def getMaxIX(d,iz):
    output = np.ndarray(iz.shape)
    for k in d:
        output[iz == k] = d[k]
    return output

##############################################
#          ix shifting transformations       #
# to account for wrap around in ix numbering #
##############################################

# since most events in our samples are in modules 0,1,11, apply an offset to center range on module 0
# module number counting is also in the opposite direction as ix.. 
# use -module instead so neighboring cells remain continuous
def getModuleShifted(module):
    return np.remainder(18 - module, np.ones_like(module)*12)

# absoluteIX index, taking into account module number
# shifted to always be non-negative
def getAbsoluteIXECAL(ix,iz,module):
    module_shifted = getModuleShifted(module)
    max_ix = getMaxIX(max_ix_perlayer_ECAL,iz)
    ncells_module = 1 + (2 * max_ix)
    return ix + max_ix + module_shifted*ncells_module

# absoluteIX index, taking into account module number
# shifted to always be non-negative
def getAbsoluteIXHCAL(ix,iz,module):
    module_shifted = getModuleShifted(module)
    max_ix = getMaxIX(max_ix_perlayer_HCAL,iz)
    ncells_module = 1 + (2 * max_ix)
    return ix + max_ix + module_shifted*ncells_module

# inverse transformation of getAbsoluteIXECAL
def invertAbsoluteIXECAL(absoluteIX,iz,module):
    module_shifted = getModuleShifted(module)
    max_ix = getMaxIX(max_ix_perlayer_ECAL,iz)
    ncells_module = 1 + (2 * max_ix)
    return absoluteIX - max_ix - module_shifted*ncells_module

# inverse transformation of getAbsoluteIXHCAL
def invertAbsoluteIXHCAL(absoluteIX,iz,module):
    module_shifted = getModuleShifted(module)
    max_ix = getMaxIX(max_ix_perlayer_HCAL,iz)
    ncells_module = 1 + (2 * max_ix)
    return absoluteIX - max_ix - module_shifted*ncells_module

####################################################################################
#                          ix rescaling transformations                            #
# to account for different granularity in different depth layers, and ECAL vs HCAL #
####################################################################################

# convert ECAL ix for later layers to look like innermost layer
def getTransformedIXECAL(absoluteIX,iz):
    max_ix = getMaxIX(max_ix_perlayer_ECAL,iz)
    ncells_module = 1 + (2 * max_ix)
    ncells_innerECAL = 1 + (2 * max_ix_perlayer_ECAL[0])
    return np.rint(ncells_innerECAL/ncells_module*absoluteIX)
    
# convert HCAL ix roughly to ECAL ix for barycenter calculation
def getTransformedIXHCAL(absoluteIX,iz):
    max_ix = getMaxIX(max_ix_perlayer_HCAL,iz)
    ncells_module = 1 + (2 * max_ix)
    ncells_innerECAL = 1 + (2 * max_ix_perlayer_ECAL[0])
    return np.rint(ncells_innerECAL/ncells_module*absoluteIX)

# convert ECAL innermost layer ix roughly to later layers for cell selection
def invertTransformedIXECAL(transformedIX,iz):
    max_ix = getMaxIX(max_ix_perlayer_ECAL,iz)
    ncells_module = 1 + (2 * max_ix)
    ncells_innerECAL = 1 + (2 * max_ix_perlayer_ECAL[0])
    return np.rint(ncells_module/ncells_innerECAL*transformedIX)

# ECAL ix roughly back to HCAL for cell selection
def invertTransformedIXHCAL(transformedIX,iz):
    max_ix = getMaxIX(max_ix_perlayer_HCAL,iz)
    ncells_module = 1 + (2 * max_ix)
    ncells_innerECAL = 1 + (2 * max_ix_perlayer_ECAL[0])
    return np.rint(ncells_module/ncells_innerECAL*transformedIX)

############################################
#      iy rescaling transformations        #
# to account for wider cells in iy in HCAL #
############################################

# cells are 6x larger in iy in HCAL compared to ECAL
def getTransformedIYHCAL(iy):
    return iy*6

# cells are 6x larger in iy in HCAL compared to ECAL
def invertTransformedIYHCAL(iy):
    return iy/6

##################################
# Functions for finding centroid #
##################################

# Given a list of distances and corresponding weights, calculate the weighted average
def findMidpoint(distance, energy):
    return np.average(distance, weights = energy)
    
# Given an array of cell hits (ix,iy,iz,E,X,Y,Z,module,absoluteIX,transformedIX,transformedIY), 
# returns the weighted average (transformed ix, transformed iy)
def findEventMidpoint(event):
    ix_avg = findMidpoint(event[:,9], event[:,3]) 
    iy_avg = findMidpoint(event[:,10], event[:,3]) 
    # don't round yet, to better center each layer and ECAL vs HCAL
    return (ix_avg,iy_avg)

# Given an array of cell hits (ix,iy,iz,E,X,Y,Z,module,absoluteIX,transformedIX,transformedIY), 
# returns the weighted average in global coords (X, Y, Z)
def findGlobalBarycenter(event):
    x_avg = findMidpoint(event[:,4], event[:,3]) 
    y_avg = findMidpoint(event[:,5], event[:,3]) 
    z_avg = findMidpoint(event[:,6], event[:,3]) 
    return (x_avg,y_avg,z_avg)

###########################################
# Cartesian to sperhical coord conversion #
###########################################

def getTheta(x, y, z):
    return math.acos(z / math.sqrt(x*x+y*y+z*z))

def getEta(theta):
    return -math.log(math.tan(theta/2.0))

def getPhi(x, y, _ = None):
    return math.atan(y/x)

##################################################
# Check if between bounds for calorimeter window #
##################################################

#Checking range for calo window (including min and max as the total number of cells is odd)
# FIXME: this assumes that wrap-around isn't a problem, after shifting to absoluteIX
def withinWindow(value, mymin, mymax):
    return (value >= mymin and value <= mymax)

########################################
# Implementation of calorimeter window #
########################################

# Given an event and local (transformedIX, transformedIY) coordinates of the calo barycenter
# get the 51x51x25 array of ECAL energies around its barycentre
def getECALArray(event,ix_avg,iy_avg):

    # get equivalent of average absolute ix in each depth layer
    ix_avg_layers = invertTransformedIXECAL(ix_avg,event[:,2])

    event_avgs = np.concatenate((event,np.reshape(ix_avg_layers,(-1,1))),axis=1)

    # Get the limit points for our grid in iy. ix depends on the layer
    iy_min = round(iy_avg) - 25
    iy_max = round(iy_avg) + 25
    
    # Create the empty array to put the energies in
    # CHECKPOINT - cells are non-uniform in z after 16 layers (last layers are twice as thick, according to https://www.dropbox.com/s/ktu1ly0ge9n4jyd/CaloImagingDataset.pdf?dl=0)
    final_array = np.zeros((51, 51, 25))
    
    # Fill the array with energy values, if they exist
    for ix, iy, iz, E, x, y, z, module, absIX, transIX, transIY, avgIX in event_avgs:
        ix_min = avgIX - 25
        ix_max = avgIX + 25
        # FIXME: this assumes that wrap-around isn't a problem, after shifting to absoluteIX
        if withinWindow(absIX, ix_min, ix_max) and withinWindow(iy, iy_min, iy_max):
            final_array[int(absIX-ix_min),int(iy-iy_min),int(iz)] = E
    return final_array

# Given an event and local (transformedIX, transformedIY) coordinates of the calo barycenter
# get the 11x11x60 array of energies around the same coordinates of HCAL
def getHCALArray(event,ix_avg,iy_avg):
    
    # get equivalent of average absolute ix in each depth layer
    ix_avg_layers = invertTransformedIXHCAL(ix_avg,event[:,2])

    event_avgs = np.concatenate((event,np.reshape(ix_avg_layers,(-1,1))),axis=1)

    # Get the limit points for our grid in iy. ix depends on the layer
    iy_avg_HCAL = round(invertTransformedIYHCAL(iy_avg))
    iy_min = iy_avg_HCAL - 5
    iy_max = iy_avg_HCAL + 5
    
    # Create the empty array to put the energies in
    final_array = np.zeros((11, 11, 60))
    
    # Fill the array with energy values, if they exist
    for ix, iy, iz, E, x, y, z, module, absIX, transIX, transIY, avgIX in event_avgs:
        ix_min = avgIX - 5
        ix_max = avgIX + 5
        # FIXME: this assumes that wrap-around isn't a problem, after shifting to absoluteIX
        if withinWindow(absIX, ix_min, ix_max) and withinWindow(iy, iy_min, iy_max):
            final_array[int(absIX-ix_min),int(iy-iy_min),int(iz)] = E
    return final_array

############################
# File reading and writing #
############################

def convertFile(inFile, outFile):

    myFeatures = FeaturesList()

    # Open file and extract events
    with open(inFile) as myfile:
        # loop line by line to limit memory usage
        for index,line in enumerate(myfile):
            my_event_string = line.replace('\n', '')
            my_event_string = my_event_string.replace(' ', '')
            my_event_string = my_event_string.replace('}{','} {')

            ########################
            # Calculating features #
            ########################

            my_event = ast.literal_eval(my_event_string)

            if index%200 == 0:
                print "Event", index

            # Make a list containing all the cell readouts of ECAL for the event and store it in a single ECAL array
            ECAL_list = []
            for cell_readout in my_event['ECAL']:
                ECAL_list.append(np.array(cell_readout))

            # Make a list containing all the cell readouts of HCAL for the event and store it in a single HCAL array
            HCAL_list = []
            for cell_reading in my_event['HCAL']:
                HCAL_list.append(np.array(cell_reading))

            # check that either ECAL or HCAL has at least one hit
            if len(ECAL_list) <= 0 and len(HCAL_list) <= 0: continue

            # create dummy array if empty
            if len(ECAL_list) == 0:
                ECAL_list.append(np.zeros(8))
            if len(HCAL_list) == 0:
                HCAL_list.append(np.zeros(8))

            # convert lists to arrays
            ECAL_array = np.array(ECAL_list)
            HCAL_array = np.array(HCAL_list)

            # compute absolute and transformed ix coordinates for window centering and selection
            # "absolute" ix accounts for the wrap-around in phi in each of 12 phi modules
            absoluteIX_ECAL = getAbsoluteIXECAL(ECAL_array[:,0],ECAL_array[:,2],ECAL_array[:,7])
            absoluteIX_HCAL = getAbsoluteIXHCAL(HCAL_array[:,0],HCAL_array[:,2],HCAL_array[:,7])
            # "transformed" ix further accounts for different cell size in phi as a function of layer depth, and ECAL vs HCAL
            transformedIX_ECAL = getTransformedIXECAL(absoluteIX_ECAL,ECAL_array[:,2])
            transformedIX_HCAL = getTransformedIXHCAL(absoluteIX_HCAL,HCAL_array[:,2])
            # "transformed" iy accounts for different cell size iy for ECAL vs HCAL
            transformedIY_ECAL = ECAL_array[:,1] # no tranformation applied to ECAL
            transformedIY_HCAL = getTransformedIYHCAL(HCAL_array[:,1])

            # add extra info to arrays
            ECAL_array_extra = np.concatenate( (ECAL_array,
                                                np.reshape(absoluteIX_ECAL,(-1,1)),
                                                np.reshape(transformedIX_ECAL,(-1,1)),
                                                np.reshape(transformedIY_ECAL,(-1,1)) ), 
                                               axis=1 )
            HCAL_array_extra = np.concatenate( (HCAL_array,
                                                np.reshape(absoluteIX_HCAL,(-1,1)),
                                                np.reshape(transformedIX_HCAL,(-1,1)),
                                                np.reshape(transformedIY_HCAL,(-1,1)) ), 
                                               axis=1 )

            # get barycenter taking into account ECAL and HCAL
            fullcalo_array = np.concatenate((ECAL_array_extra,HCAL_array_extra),axis=0)
            # returns the midpoint in (transformed ix, transformed iy) local coordinates
            midpoint_local = findEventMidpoint(fullcalo_array)

            # returns ECAL 51x51x25 array around barycenter based on midpoint in (transformed ix, transformed iy) local coordinates
            ECAL_window = getECALArray(ECAL_array_extra,midpoint_local[0],midpoint_local[1])/1000.*50 # Geant is in units of 1/50 GeV for some reason
            myFeatures.add("ECAL", ECAL_window)

            # returns HCAL 11x11x60 array around barycenter based on midpoint in (transformed ix, transformed iy) local coordinates
            HCAL_window = getHCALArray(HCAL_array_extra,midpoint_local[0],midpoint_local[1])/1000.*50 # Geant is in units of 1/50 GeV for some reason
            myFeatures.add("HCAL", HCAL_window)
            
            # get barycenter in global coords to compute barycenter theta, eta, phi
            barycenter_global = findGlobalBarycenter(fullcalo_array)
            theta = getTheta(*barycenter_global)
            eta = getEta(theta)
            phi = getPhi(*barycenter_global)
            myFeatures.add("recoTheta", theta)
            myFeatures.add("recoEta", eta)
            myFeatures.add("recoPhi", phi)

            # Truth info from txt file
            myFeatures.add("energy", my_event['E']/1000.) # convert MeV to GeV
            myFeatures.add("pdgID", my_event['pdgID'])
            myFeatures.add("conversion", my_event['conversion'])
            myFeatures.add("openingAngle", my_event['openingAngle'])

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
    print "Converting file"
    convertFile(inFile, outFile+"_temp")
    print "Calculating features"
    addFeatures.convertFile(outFile+"_temp", outFile)
    os.remove(outFile+"_temp")
