import h5py as h5
import numpy as np
from math import ceil, floor
import sys, pdb

def resample_3D(ECAL, x_scale, y_scale, z_scale):
    # resample hit info for new cell sizes
    (x, y, z) = ECAL.shape
    new_x = int(ceil(x/x_scale)) # minimum number of new-sized cells that can encompass the hit info
    x_overhang = 0 if x%x_scale==0 else (x_scale - (x%x_scale)) / 2
    new_y = int(ceil(y/y_scale))
    y_overhang = 0 if y%y_scale==0 else (y_scale - (y%y_scale)) / 2
    new_z = int(ceil(z/z_scale))
    z_overhang = 0 if z%z_scale==0 else (z_scale - (z%z_scale)) / 2
    new_ECAL = np.zeros((new_x, new_y, new_z))
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
                                new_ECAL[x_j][y_j][z_j] += cell_fraction*ECAL[x_i][y_i][z_i]
    ECAL = new_ECAL
    new_ECAL = np.zeros((x, y, z))
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
                                new_ECAL[x_i][y_i][z_i] += cell_fraction*ECAL[x_j][y_j][z_j]

def spoof_ATLAS_geometry(ECAL):
    # geometry taken from https://indico.cern.ch/event/693870/contributions/2890799/attachments/1597965/2532270/CaloMeeting-Feb-09-18.pdf
    new_ECAL = np.empty_like(ECAL)
    x, y, z = ECAL.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                pass
    return new_ECAL

def spoof_CMS_geometry(ECAL):
    # geometry taken from https://indico.cern.ch/event/693870/contributions/2890799/attachments/1597965/2532270/CaloMeeting-Feb-09-18.pdf
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
    z_scale = ECAL.shape[2] # just collapse the whole thing
    new_ECAL = resample_3D(ECAL, x_scale, y_scale, z_scale)
    return new_ECAL

#################
# MAIN FUNCTION #
#################

if __name__ == "__main__":

    in_file_path = sys.argv[1]
    out_file_path = sys.argv[2]
    resample_type = int(sys.argv[3])

    in_file = h5.File(in_file_path)
    print("Working on converting file " + in_file_path)

    ECAL = in_file['ECAL']
    new_ECAL = []
    for i, ECAL_event in enumerate(ECAL):
        if (i%1000 == 0):
            print(str(i), "out of", len(ECAL))
        if resample_type == 0:
            new_ECAL_event = spoof_ATLAS_geometry(ECAL_event)
        elif resample_type == 1:
            new_ECAL_event = spoof_CMS_geometry(ECAL_event)
        new_ECAL.append(new_ECAL_event)
    out_file = h5.File(out_file_path, "w")
    out_file.create_dataset('ECAL', data=np.array(new_ECAL).squeeze(), compression='gzip')

    keep_features = ['HCAL', 'recoTheta', 'recoEta', 'recoPhi', 'energy', 'pdgID', 'conversion', 'openingAngle']
    for feature in keep_features:
        out_file.create_dataset(feature, data=in_file[feature], compression='gzip')
    print("Finished converting file " + out_file_path)
