import h5py as h5
import numpy as np
from math import ceil, floor
import pdb
# import matplotlib.pyplot as plt

# def plot_ECAL(ECAL, save_name):
    # x,y,z = ECAL.nonzero() 
    # fig = plt.figure() 
    # ax = fig.add_subplot(111, projection='3d') 
    # ax.scatter(x, y, -z, marker='.', zdir='z', c=ECAL[x,y,z], cmap='jet', alpha=0.3) 
    # ax.set_xlabel('X') 
    # ax.set_ylabel('Y') 
    # ax.set_zlabel('Z') 
    # plt.savefig(save_name)

def resample_3D(ECAL, x_scale, y_scale, z_scale):
    print("OLD")
    print(ECAL)
    # resample hit info for new cell sizes
    (z, x, y) = ECAL.shape
    new_x = int(ceil(x/x_scale)) # minimum number of new-sized cells that can encompass the hit info
    x_overhang = 0 if x%x_scale==0 else (x_scale - (x%x_scale)) / 2
    new_y = int(ceil(y/y_scale))
    y_overhang = 0 if y%y_scale==0 else (y_scale - (y%y_scale)) / 2
    new_z = int(ceil(z/z_scale))
    z_overhang = 0 if z%z_scale==0 else (z_scale - (z%z_scale)) / 2
    new_ECAL = np.zeros((new_z, new_x, new_y))
    for x_i in range(x):
        for y_i in range(y):
            for z_i in range(z):
                low_x = int(floor((x_i + x_overhang) / x_scale)) # which new bin the bottom of this cell should go in
                low_y = int(floor((y_i + y_overhang) / y_scale))
                low_z = int(floor((z_i + z_overhang) / z_scale))
                fraction_low_x = min(x_scale - ((x_i + x_overhang) % x_scale), 1) # what fraction of this cell goes in the low bin
                fraction_low_y = min(y_scale - ((y_i + y_overhang) % y_scale), 1)
                fraction_low_z = min(z_scale - ((z_i + z_overhang) % z_scale), 1)
                high_x = int(floor((x_i + 1 + x_overhang) / x_scale)) # highest bin this cell will go into
                high_y = int(floor((y_i + 1 + y_overhang) / y_scale))
                high_z = int(floor((z_i + 1 + z_overhang) / z_scale))
                fraction_high_x = min((x_i + 1 + x_overhang) % x_scale, 1) # what fraction of this cell goes in the high bin
                fraction_high_y = min((y_i + 1 + y_overhang) % y_scale, 1)
                fraction_high_z = min((z_i + 1 + z_overhang) % z_scale, 1)
                fraction_middle_x = 1/x_scale # just in case there's intermediate bins
                fraction_middle_y = 1/y_scale # just in case there's intermediate bins
                fraction_middle_z = 1/z_scale # just in case there's intermediate bins
                for x_j in range(low_x, high_x+1):
                    for y_j in range(low_y, high_y+1):
                        for z_j in range(low_z, high_z+1):
                            x_fraction = fraction_middle_x
                            y_fraction = fraction_middle_y
                            z_fraction = fraction_middle_z
                            if x_j == low_x: x_fraction = fraction_low_x
                            if y_j == low_y: y_fraction = fraction_low_y
                            if z_j == low_z: z_fraction = fraction_low_z
                            if x_j == high_x: x_fraction = fraction_high_x
                            if y_j == high_y: y_fraction = fraction_high_y
                            if z_j == high_z: z_fraction = fraction_high_z
                            cell_fraction = x_fraction*y_fraction*z_fraction
                            if cell_fraction != 0:
                                new_ECAL[z_j][x_j][y_j] += cell_fraction*ECAL[z_i][x_i][y_i]
    print("RESAMPLING")
    print(new_ECAL)
    ECAL = new_ECAL
    new_ECAL = np.zeros((z, x, y))
    # resample back to old cell sizes
    for x_i in range(x):
        for y_i in range(y):
            for z_i in range(z):
                low_x = int(floor((x_i + x_overhang) / x_scale)) # which new bin the bottom of this cell should go in
                low_y = int(floor((y_i + y_overhang) / y_scale))
                low_z = int(floor((z_i + z_overhang) / z_scale))
                fraction_low_x = min(1 - ((x_i + x_overhang) % x_scale) / x_scale, 1) # what fraction of this cell goes in the low bin
                fraction_low_y = min(1 - ((y_i + y_overhang) % y_scale) / y_scale, 1)
                fraction_low_z = min(1 - ((z_i + z_overhang) % z_scale) / z_scale, 1)
                high_x = int(floor((x_i + 1 + x_overhang) / x_scale)) # highest bin this cell will go into
                high_y = int(floor((y_i + 1 + y_overhang) / y_scale))
                high_z = int(floor((z_i + 1 + z_overhang) / z_scale))
                fraction_high_x = min(((x_i + 1 + x_overhang) % x_scale) / x_scale, 1) # what fraction of this cell goes in the high bin
                fraction_high_y = min(((y_i + 1 + y_overhang) % y_scale) / y_scale, 1)
                fraction_high_z = min(((z_i + 1 + z_overhang) % z_scale) / z_scale, 1)
                fraction_middle_x = 1/x_scale # just in case there's intermediate bins
                fraction_middle_y = 1/y_scale # just in case there's intermediate bins
                fraction_middle_z = 1/z_scale # just in case there's intermediate bins
                for x_j in range(low_x, high_x+1):
                    for y_j in range(low_y, high_y+1):
                        for z_j in range(low_z, high_z+1):
                            x_fraction = fraction_middle_x
                            y_fraction = fraction_middle_y
                            z_fraction = fraction_middle_z
                            if x_j == low_x: x_fraction = fraction_low_x
                            if y_j == low_y: y_fraction = fraction_low_y
                            if z_j == low_z: z_fraction = fraction_low_z
                            if x_j == high_x: x_fraction = fraction_high_x
                            if y_j == high_y: y_fraction = fraction_high_y
                            if z_j == high_z: z_fraction = fraction_high_z
                            cell_fraction = x_fraction*y_fraction*z_fraction
                            if cell_fraction != 0:
                                new_ECAL[z_i][x_i][y_i] += cell_fraction*ECAL[z_j][x_j][y_j]
    print("NEW")
    print(new_ECAL)

def spoof_ATLAS_geometry(ECAL):
    # geometry conversion
    new_ECAL = np.empty_like(ECAL)
    x, y, z = ECAL.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                pass
    return new_ECAL

def spoof_CMS_geometry(ECAL):
    # CLIC and CMS values
    CLIC_Moliere = 0.9327 # cm
    CMS_Moliere = 1.959
    CLIC_radiation = 0.3504
    CMS_radiation = 0.8903 
    CLIC_eta = 0.003
    CLIC_phi = 0.003
    CMS_eta = 0.0175
    CMS_phi = 0.0175
    # resample to match CMS
    x_scale = (CMS_eta/CMS_Moliere) / (CLIC_eta/CLIC_Moliere) # how many cells go together to form a new cell
    y_scale = x_scale
    z_scale = z # just collapse the whole thing
    new_ECAL = resample_3D(ECAL, x_scale, y_scale, z_scale)
    return new_ECAL

#################
# MAIN FUNCTION #
#################

if __name__ == "__main__":
    ECAL = np.array([[[0, 1, 1, 0, 0],
            [1, 3, 2, 1, 0],
            [1, 3, 4, 2, 1],
            [2, 3, 4, 3, 3],
            [0, 1, 2, 3, 3]]])
    resample_3D(ECAL, 1.5, 1.5, 1)
    # data = h5.File("/data/LCD/NewSamples/RandomAngle/EleEscan_RandomAngle_MERGED/EleEscan_RandomAngle_1_1.h5")
    # ECAL = data['ECAL']
    # for i, ECAL_event in enumerate(ECAL):
        # if i > 10:
            # break
        # # plot_ECAL(ECAL_event, "Plots/ECAL_"+str(i)+"_before.png")
        # new_ECAL = spoof_CMS_geometry(ECAL_event)
        # # plot_ECAL(new_ECAL, "Plots/ECAL_"+str(i)+"_after.png")
