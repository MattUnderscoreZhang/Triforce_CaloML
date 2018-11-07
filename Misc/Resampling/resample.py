from __future__ import division
import h5py as h5
import numpy as np
from math import ceil, floor
import sys, pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############
# PLOTTING #
############

def plot_ECAL(ECAL, save_name):
    x,y,z = ECAL.nonzero() 
    fig = plt.figure() 
    ax = Axes3D(fig)
    ax.scatter(x, y, -z, marker='.', zdir='z', c=ECAL[x,y,z], cmap='jet', alpha=0.3) 
    ax.set_xlabel('X') 
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') 
    plt.savefig(save_name)

##############
# RESAMPLING #
##############

def get_resampling_matrix(ECAL_x, ECAL_y, x_scale, y_scale):
    # x_scale and y_scale are how many old cells go together to form a new cell
    new_x = int(ceil(ECAL_x/x_scale)) # minimum number of new-sized cells that can encompass the hit info
    x_overhang = 0 if ECAL_x%x_scale==0 else (x_scale - (ECAL_x%x_scale)) / 2
    new_y = int(ceil(ECAL_y/y_scale))
    y_overhang = 0 if ECAL_y%y_scale==0 else (y_scale - (ECAL_y%y_scale)) / 2
    # first resampling
    first_resampling_matrix = np.zeros((ECAL_x*ECAL_y, new_x*new_y))
    for x_i in range(ECAL_x):
        for y_i in range(ECAL_y):
            low_x = int(floor((x_i + x_overhang) / x_scale)) # which new bin the bottom of this cell should go in
            low_y = int(floor((y_i + y_overhang) / y_scale))
            fraction_low_x = min(x_scale - ((x_i + x_overhang) % x_scale), 1) # what fraction of this cell goes in the low bin
            fraction_low_y = min(y_scale - ((y_i + y_overhang) % y_scale), 1)
            high_x = int(floor((x_i + 1 + x_overhang) / x_scale)) # highest bin this cell will go into
            high_y = int(floor((y_i + 1 + y_overhang) / y_scale))
            fraction_high_x = min((x_i + 1 + x_overhang) % x_scale, 1) # what fraction of this cell goes in the high bin
            fraction_high_y = min((y_i + 1 + y_overhang) % y_scale, 1)
            fraction_middle_x = 1/x_scale # just in case there's intermediate bins
            fraction_middle_y = 1/y_scale # just in case there's intermediate bins
            for x_j in range(low_x, high_x+1):
                for y_j in range(low_y, high_y+1):
                    x_fraction = fraction_middle_x
                    y_fraction = fraction_middle_y
                    if x_j == low_x: x_fraction = fraction_low_x
                    if y_j == low_y: y_fraction = fraction_low_y
                    if x_j == high_x: x_fraction = fraction_high_x
                    if y_j == high_y: y_fraction = fraction_high_y
                    cell_fraction = x_fraction*y_fraction
                    if cell_fraction != 0:
                        first_resampling_matrix[x_i*ECAL_y+y_i, x_j*new_y+y_j] += cell_fraction
    # resample back to old cell sizes
    second_resampling_matrix = np.zeros((new_x*new_y, ECAL_x*ECAL_y))
    for x_i in range(ECAL_x):
        for y_i in range(ECAL_y):
            low_x = int(floor((x_i + x_overhang) / x_scale)) # which new bin the bottom of this cell should go in
            low_y = int(floor((y_i + y_overhang) / y_scale))
            fraction_low_x = min(1 - ((x_i + x_overhang) % x_scale) / x_scale, 1) # what fraction of this cell goes in the low bin
            fraction_low_y = min(1 - ((y_i + y_overhang) % y_scale) / y_scale, 1)
            high_x = int(floor((x_i + 1 + x_overhang) / x_scale)) # highest bin this cell will go into
            high_y = int(floor((y_i + 1 + y_overhang) / y_scale))
            fraction_high_x = min(((x_i + 1 + x_overhang) % x_scale) / x_scale, 1) # what fraction of this cell goes in the high bin
            fraction_high_y = min(((y_i + 1 + y_overhang) % y_scale) / y_scale, 1)
            fraction_middle_x = 1/x_scale # just in case there's intermediate bins
            fraction_middle_y = 1/y_scale # just in case there's intermediate bins
            for x_j in range(low_x, high_x+1):
                for y_j in range(low_y, high_y+1):
                    x_fraction = fraction_middle_x
                    y_fraction = fraction_middle_y
                    if x_j == low_x: x_fraction = fraction_low_x
                    if y_j == low_y: y_fraction = fraction_low_y
                    if x_j == high_x: x_fraction = fraction_high_x
                    if y_j == high_y: y_fraction = fraction_high_y
                    cell_fraction = x_fraction*y_fraction
                    if cell_fraction != 0:
                        second_resampling_matrix[x_j*new_y+y_j, x_i*ECAL_y+y_i] += cell_fraction
    return np.matmul(first_resampling_matrix, second_resampling_matrix)

def resample_2D(ECAL, resampling_matrix):
    # unroll into vector
    ECAL_x = ECAL.shape[0]
    ECAL_y = ECAL.shape[1]
    ECAL = ECAL.reshape((ECAL_x * ECAL_y, 1))
    ECAL = np.matmul(resampling_matrix, ECAL)
    ECAL = ECAL.reshape((ECAL_x, ECAL_y))
    return ECAL

#########
# ATLAS #
#########

def get_ATLAS_resampling_matrices(ECAL_shape):
    # geometry taken from https://indico.cern.ch/event/693870/contributions/2890799/attachments/1597965/2532270/CaloMeeting-Feb-09-18.pdf
    CLIC_Moliere = 0.9327 # cm
    CLIC_radiation = 0.3504
    CLIC_eta = 0.003
    CLIC_phi = 0.003
    ATLAS_Moliere = 9.043
    ATLAS_radiation = 14
    ECAL_x = ECAL_shape[0]
    ECAL_y = ECAL_shape[1]
    # layer 1
    ATLAS_eta = 0.025/8
    ATLAS_phi = 0.1
    x_scale = (ATLAS_eta/ATLAS_Moliere) / (CLIC_eta/CLIC_Moliere) # how many cells go together to form a new cell
    y_scale = (ATLAS_phi/ATLAS_Moliere) / (CLIC_phi/CLIC_Moliere)
    matrix_1 = get_resampling_matrix(ECAL_x, ECAL_y, x_scale, y_scale)
    # layer 2
    ATLAS_eta = 0.025
    ATLAS_phi = 0.025
    x_scale = (ATLAS_eta/ATLAS_Moliere) / (CLIC_eta/CLIC_Moliere)
    y_scale = (ATLAS_phi/ATLAS_Moliere) / (CLIC_phi/CLIC_Moliere)
    matrix_2 = get_resampling_matrix(ECAL_x, ECAL_y, x_scale, y_scale)
    # layer 3
    ATLAS_eta = 0.5
    ATLAS_phi = 0.025
    x_scale = (ATLAS_eta/ATLAS_Moliere) / (CLIC_eta/CLIC_Moliere)
    y_scale = (ATLAS_phi/ATLAS_Moliere) / (CLIC_phi/CLIC_Moliere)
    matrix_3 = get_resampling_matrix(ECAL_x, ECAL_y, x_scale, y_scale)
    return (matrix_1, matrix_2, matrix_3)

def spoof_ATLAS_geometry(ECAL, resampling_matrices):
    # separate in Z - layers 1:2:3 in 4.3:16:2 ratio
    # for ease, we'll split 5:17:3
    # layer 1
    new_ECAL_1 = ECAL[:,:,:5].sum(axis=2)
    new_ECAL_1 = resample_2D(new_ECAL_1, resampling_matrices[0])
    new_ECAL_1 = np.repeat(new_ECAL_1[:, :, np.newaxis]/5, 5, axis=2)
    # layer 2
    new_ECAL_2 = ECAL[:,:,5:22].sum(axis=2)
    new_ECAL_2 = resample_2D(new_ECAL_2, resampling_matrices[1])
    new_ECAL_2 = np.repeat(new_ECAL_2[:, :, np.newaxis]/17, 17, axis=2)
    # layer 3
    new_ECAL_3 = ECAL[:,:,22:25].sum(axis=2)
    new_ECAL_3 = resample_2D(new_ECAL_3, resampling_matrices[2])
    new_ECAL_3 = np.repeat(new_ECAL_3[:, :, np.newaxis]/3, 3, axis=2)
    # recombine Z
    new_ECAL = np.concatenate((new_ECAL_1, new_ECAL_2, new_ECAL_3), axis=2)
    return new_ECAL

#######
# CMS #
#######

def get_CMS_resampling_matrix(ECAL_shape):
    # geometry taken from https://indico.cern.ch/event/693870/contributions/2890799/attachments/1597965/2532270/CaloMeeting-Feb-09-18.pdf
    CLIC_Moliere = 0.9327 # cm
    CMS_Moliere = 1.959
    CLIC_radiation = 0.3504
    CMS_radiation = 0.8903 
    CLIC_eta = 0.003
    CLIC_phi = 0.003
    CMS_eta = 0.0175
    CMS_phi = 0.0175
    # calculate resampling matrix
    ECAL_x = ECAL_shape[0]
    ECAL_y = ECAL_shape[1]
    x_scale = (CMS_eta/CMS_Moliere) / (CLIC_eta/CLIC_Moliere) # how many cells go together to form a new cell
    y_scale = x_scale
    return get_resampling_matrix(ECAL_x, ECAL_y, x_scale, y_scale)

def spoof_CMS_geometry(ECAL, resampling_matrix):
    # collapse Z
    new_ECAL = ECAL.sum(axis=2)
    # resample to match CMS
    new_ECAL = resample_2D(new_ECAL, resampling_matrix)
    # uncollapse Z
    ECAL_z = ECAL.shape[2]
    new_ECAL = new_ECAL/ECAL_z
    new_ECAL = np.repeat(new_ECAL[:, :, np.newaxis], ECAL_z, axis=2)
    return new_ECAL

#################
# MAIN FUNCTION #
#################

if __name__ == "__main__":

    in_file_path = sys.argv[1]
    out_file_path = sys.argv[2]
    resample_type = int(sys.argv[3])
    make_plots = False

    in_file = h5.File(in_file_path)
    print("Working on converting file " + in_file_path)

    ECAL = in_file['ECAL'][()]
    ATLAS_resampling_matrices = get_ATLAS_resampling_matrices(ECAL[0].shape)
    CMS_resampling_matrix = get_CMS_resampling_matrix(ECAL[0].shape)

    new_ECAL = []
    for i, ECAL_event in enumerate(ECAL):
        if (i%1000 == 0):
            print(str(i), "out of", len(ECAL))
        if resample_type == 0:
            new_ECAL_event = spoof_ATLAS_geometry(ECAL_event, ATLAS_resampling_matrices)
        elif resample_type == 1:
            new_ECAL_event = spoof_CMS_geometry(ECAL_event, CMS_resampling_matrix)
        new_ECAL.append(new_ECAL_event)
        if make_plots:
            plot_ECAL(ECAL_event, "ResamplingTestCode/Plots/ECAL_"+str(i)+"_before.png")
            plot_ECAL(new_ECAL_event, "ResamplingTestCode/Plots/ECAL_"+str(i)+"_after.png")
            if i>4: sys.exit(0)
    out_file = h5.File(out_file_path, "w")
    out_file.create_dataset('ECAL', data=np.array(new_ECAL).squeeze(), compression='gzip')

    keep_features = ['HCAL', 'recoTheta', 'recoEta', 'recoPhi', 'energy', 'pdgID', 'conversion', 'openingAngle']
    for feature in keep_features:
        out_file.create_dataset(feature, data=in_file[feature], compression='gzip')
    print("Finished converting file " + out_file_path)
