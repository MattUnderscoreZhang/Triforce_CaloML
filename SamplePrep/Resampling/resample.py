from __future__ import division
import h5py as h5
import numpy as np
from math import ceil, floor
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


############
# PLOTTING #
############

def plot_ECAL(ECAL, save_name):
    x, y, z = ECAL.nonzero()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, -z, marker='.', zdir='z', c=ECAL[x, y, z], cmap='jet', alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(save_name)


########################
# RESAMPLING ALGORITHM #
########################

def get_smallest_new_size(old_size, scaling):
    '''Given an <old_size> and <scaling> tuple, find the smallest <new_size> tuple that overhangs the
       old one.'''
    return [int(ceil(x_size/x_scale)) for x_size, x_scale in zip(old_size, scaling)]


def get_largest_new_size(old_size, scaling):
    '''Given an <old_size> and <scaling> tuple, find the largest <new_size> tuple that does not overhang
       the old one.'''
    return [int(floor(x_size/x_scale)) for x_size, x_scale in zip(old_size, scaling)]


def calculate_overhang(old_size, scaling, new_size):
    '''Given an old matrix size, a dimensional scaling, and the new size, we return how many cells
       (in old matrix units) overhang on each side.'''
    return [(x_new_size*x_scale - x_old_size)/2 for x_old_size, x_scale, x_new_size in zip(old_size, scaling, new_size)]


def get_cell_filling(old_index, x_scaling, x_overhang):
    '''What is the first new cell that gets filled by old cell number <old_index>? What is the last one?
       What fraction of the old cell goes into the first cell, the last cell, and any intermediate cells?'''
    low_new_index = int(floor((old_index+x_overhang)/x_scaling))
    high_new_index = int(floor((old_index+x_overhang+1)/x_scaling))
    low_index_upper_endpoint = (low_new_index+1)*x_scaling-x_overhang
    low_frac = 1.0 if low_index_upper_endpoint >= old_index+1 else low_index_upper_endpoint % 1
    high_index_lower_endpoint = (high_new_index)*x_scaling-x_overhang
    high_frac = 1.0 if high_index_lower_endpoint <= old_index else (1 - (high_index_lower_endpoint % 1)) % 1
    if high_frac == 0.0:
        high_new_index -= 1
        high_index_lower_endpoint = (high_new_index)*x_scaling-x_overhang
        high_frac = 1.0 if high_index_lower_endpoint <= old_index else (1 - (high_index_lower_endpoint % 1)) % 1
    mid_frac = min(x_scaling, 1)
    return low_new_index, high_new_index, low_frac, high_frac, mid_frac


def one_d_resampling(x_old, x_new, x_scaling, x_overhang):
    '''Given <x_old> cells which must be distributed among <x_new> cells, where <x_scaling> old cells
       equal the size of a new cell, and the new cells overhang the old cells by <x_overhang> old cell
       widths on each side, what does the resampling behavior look like?'''
    resampling_behavior = []
    for old_index in range(x_old):
        resampling_behavior.append(get_cell_filling(old_index, x_scaling, x_overhang))
    return resampling_behavior


def get_flattened_index(matrix_shape, indices):
    '''Given a matrix of some shape and an index for a cell in the matrix, determine the flattened index.
       For example, in a matrix of size 5x7, cell (3, 4) would have index 3*7+4=25.'''
    index = 0
    for dimension in range(len(matrix_shape)):
        index += indices[dimension] * np.prod(matrix_shape[dimension+1:])
    return int(index)


def get_resampling_fill_list(old_index, dimension_resamplers, new_size):
    '''Given the index of an old cell, determine all the new cells that it will go to fill, as well as
       the fraction of the old cell which will go into each. In the form of [(new_index, fill_fraction),
       ...]'''
    resampling_dimensional_data = []
    for x_old_index, resampler in zip(old_index, dimension_resamplers):
        resampling_dimensional_data.append(resampler[x_old_index])
    lowest_fill_index = [i[0] for i in resampling_dimensional_data]
    fill_range = [i[1]-i[0]+1 for i in resampling_dimensional_data]
    resampling_fill_list = []
    for relative_fill_index in np.ndindex(tuple(fill_range)):
        fill_index = tuple([i+j for i, j in zip(lowest_fill_index, relative_fill_index)])
        if np.any([i >= j or i < 0 for i, j in zip(fill_index, new_size)]):
            continue
        fill_fraction = 1
        for i, r in zip(fill_index, resampling_dimensional_data):
            if i == r[0]:
                fill_fraction *= r[2]
            elif i == r[1]:
                fill_fraction *= r[3]
            else:
                fill_fraction *= r[4]
        resampling_fill_list.append((fill_index, fill_fraction))
    return resampling_fill_list


def get_partial_resampling_matrix(old_size, scaling, new_size):
    '''Suppose a matrix A of dimensions <old_size> is scaled by <scaling> such that <x_scaling> old cells
       add to create a new cell in dimension x. The new matrix is of size <new_size>. The resampling matrix
       B is returned such that flatten(A) * B = flatten(C), where C is the new matrix.'''
    overhang = calculate_overhang(old_size, scaling, new_size)
    resampling_matrix = np.zeros((np.prod(old_size), np.prod(new_size)))
    dimension_resamplers = []
    # get resampling behavior for each dimension
    for x_old_size, x_new_size, x_scaling, x_overhang in zip(old_size, new_size, scaling, overhang):
        dimension_resamplers.append(one_d_resampling(x_old_size, x_new_size, x_scaling, x_overhang))
    # fill resampling matrix
    for old_index in np.ndindex(tuple(old_size)):
        resampling_fill_list = get_resampling_fill_list(old_index, dimension_resamplers, new_size)
        old_flattened_index = get_flattened_index(old_size, old_index)
        for resampling_fill in resampling_fill_list:
            new_flattened_index = get_flattened_index(new_size, resampling_fill[0])
            resampling_matrix[old_flattened_index][new_flattened_index] += resampling_fill[1]
    return resampling_matrix


def invert_scaling(scaling):
    return tuple([1/i for i in scaling])


def get_full_resampling_matrix(old_size, scaling):
    new_size = get_smallest_new_size(old_size, scaling)
    first_resampling_matrix = get_partial_resampling_matrix(old_size, scaling, new_size)
    second_resampling_matrix = get_partial_resampling_matrix(new_size, invert_scaling(scaling), old_size)
    return np.matmul(first_resampling_matrix, second_resampling_matrix)


def resample_2D(ECAL, resampling_matrix):
    # unroll into vector
    ECAL_x = ECAL.shape[0]
    ECAL_y = ECAL.shape[1]
    ECAL = ECAL.reshape((ECAL_x * ECAL_y, 1))
    ECAL = np.matmul(resampling_matrix, ECAL)
    ECAL = ECAL.reshape((ECAL_x, ECAL_y))
    return ECAL


#######################
# RESAMPLING MATRICES #
#######################

def get_ATLAS_resampling_matrices(ECAL_shape):
    # geometry taken from https://indico.cern.ch/event/693870/contributions/2890799/attachments/1597965/2532270/CaloMeeting-Feb-09-18.pdf
    CLIC_Moliere = 0.9327  # cm
    # CLIC_radiation = 0.3504
    CLIC_eta = 0.003
    CLIC_phi = 0.003
    ATLAS_Moliere = 9.043
    # ATLAS_radiation = 14
    ECAL_x = ECAL_shape[0]
    ECAL_y = ECAL_shape[1]
    # layer 1
    ATLAS_eta = 0.025/8
    ATLAS_phi = 0.1
    x_scale = (ATLAS_eta/ATLAS_Moliere) / (CLIC_eta/CLIC_Moliere)  # how many cells go together to form a new cell
    y_scale = (ATLAS_phi/ATLAS_Moliere) / (CLIC_phi/CLIC_Moliere)
    matrix_1 = get_full_resampling_matrix((ECAL_x, ECAL_y), (x_scale, y_scale))
    # layer 2
    ATLAS_eta = 0.025
    ATLAS_phi = 0.025
    x_scale = (ATLAS_eta/ATLAS_Moliere) / (CLIC_eta/CLIC_Moliere)
    y_scale = (ATLAS_phi/ATLAS_Moliere) / (CLIC_phi/CLIC_Moliere)
    matrix_2 = get_full_resampling_matrix((ECAL_x, ECAL_y), (x_scale, y_scale))
    # layer 3
    ATLAS_eta = 0.5
    ATLAS_phi = 0.025
    x_scale = (ATLAS_eta/ATLAS_Moliere) / (CLIC_eta/CLIC_Moliere)
    y_scale = (ATLAS_phi/ATLAS_Moliere) / (CLIC_phi/CLIC_Moliere)
    matrix_3 = get_full_resampling_matrix((ECAL_x, ECAL_y), (x_scale, y_scale))
    return (matrix_1, matrix_2, matrix_3)


def get_CMS_resampling_matrix(ECAL_shape):
    # geometry taken from https://indico.cern.ch/event/693870/contributions/2890799/attachments/1597965/2532270/CaloMeeting-Feb-09-18.pdf
    CLIC_Moliere = 0.9327  # cm
    CMS_Moliere = 1.959
    # CLIC_radiation = 0.3504
    # CMS_radiation = 0.8903
    CLIC_eta = 0.003
    # CLIC_phi = 0.003
    CMS_eta = 0.0175
    # CMS_phi = 0.0175
    # calculate resampling matrix
    ECAL_x = ECAL_shape[0]
    ECAL_y = ECAL_shape[1]
    x_scale = (CMS_eta/CMS_Moliere) / (CLIC_eta/CLIC_Moliere)  # how many cells go together to form a new cell
    y_scale = x_scale
    return get_full_resampling_matrix((ECAL_x, ECAL_y), (x_scale, y_scale))


##############
# RESAMPLING #
##############

def spoof_ATLAS_geometry(ECAL, resampling_matrices):
    # separate in Z - layers 1:2:3 in 4.3:16:2 ratio
    # for ease, we'll split 5:17:3
    # layer 1
    new_ECAL_1 = ECAL[:, :, :5].sum(axis=2)
    new_ECAL_1 = resample_2D(new_ECAL_1, resampling_matrices[0])
    new_ECAL_1 = np.repeat(new_ECAL_1[:, :, np.newaxis]/5, 5, axis=2)
    # layer 2
    new_ECAL_2 = ECAL[:, :, 5:22].sum(axis=2)
    new_ECAL_2 = resample_2D(new_ECAL_2, resampling_matrices[1])
    new_ECAL_2 = np.repeat(new_ECAL_2[:, :, np.newaxis]/17, 17, axis=2)
    # layer 3
    new_ECAL_3 = ECAL[:, :, 22:25].sum(axis=2)
    new_ECAL_3 = resample_2D(new_ECAL_3, resampling_matrices[2])
    new_ECAL_3 = np.repeat(new_ECAL_3[:, :, np.newaxis]/3, 3, axis=2)
    # recombine Z
    new_ECAL = np.concatenate((new_ECAL_1, new_ECAL_2, new_ECAL_3), axis=2)
    return new_ECAL


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
    resample_type = sys.argv[3]
    make_plots = False

    in_file = h5.File(in_file_path)
    print("Working on converting file " + in_file_path)

    ECAL = in_file['ECAL'][()]
    ATLAS_resampling_matrices = get_ATLAS_resampling_matrices(ECAL[0].shape)
    CMS_resampling_matrix = get_CMS_resampling_matrix(ECAL[0].shape)

    new_ECAL = []
    for i, ECAL_event in enumerate(ECAL):
        if (i % 1000 == 0):
            print(str(i), "out of", len(ECAL))
        if resample_type == "ATLAS":
            new_ECAL_event = spoof_ATLAS_geometry(ECAL_event, ATLAS_resampling_matrices)
        elif resample_type == "CMS":
            new_ECAL_event = spoof_CMS_geometry(ECAL_event, CMS_resampling_matrix)
        new_ECAL.append(new_ECAL_event)
        if make_plots:
            plot_ECAL(ECAL_event, "ResamplingTestCode/Plots/ECAL_"+str(i)+"_before.png")
            plot_ECAL(new_ECAL_event, "ResamplingTestCode/Plots/ECAL_"+str(i)+"_after.png")
            if i > 4:
                sys.exit(0)
    out_file = h5.File(out_file_path, "w")
    out_file.create_dataset('ECAL', data=np.array(new_ECAL).squeeze(), compression='gzip')

    keep_features = ['HCAL', 'recoTheta', 'recoEta', 'recoPhi', 'energy', 'pdgID', 'conversion', 'openingAngle']
    for feature in keep_features:
        out_file.create_dataset(feature, data=in_file[feature], compression='gzip')
    print("Finished converting file " + out_file_path)
