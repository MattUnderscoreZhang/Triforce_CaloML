import sys
import numpy as np
import glob
from h5py import File
import os

path1=sys.argv[1]
path2=sys.argv[2]
num_cells=int(sys.argv[3])

files1=sorted(glob.glob(path1))

old_ecal_size = 25
merge_ecal_size=num_cells
new_ecal_size = old_ecal_size/merge_ecal_size
num_per_file = 10000

for i in range(len(files1)):
    f1=File(files1[i], 'r')
    ecal_old=f1['ECAL'][:]

    ecal_new=np.zeros((num_per_file,new_ecal_size, new_ecal_size, old_ecal_size))

    assert(ecal_new.shape[0]==num_per_file)

    for k in range(num_per_file):
        for l in range(old_ecal_size):
            for m in range(new_ecal_size):
                for n in range(new_ecal_size):
                    for o in range(merge_ecal_size):
                        for p in range(merge_ecal_size):
                            ecal_new[k][n][m][l] += ecal_old[k][merge_ecal_size*n+o][merge_ecal_size*m+p][l]
                            #ecal_new[k][n][m][l] = ecal_old[k][merge_ecal_size*n][merge_ecal_size*m][l] + ecal_old[k][merge_ecal_size*n+1][merge_ecal_size*m][l] + ecal_old[k][merge_ecal_size*n][merge_ecal_size*m+1][l] + ecal_old[k][merge_ecal_size*n+1][merge_ecal_size*m+1][l]

    keepFeatures = ['HCAL', 'Event/pdgID', 'Event/energy', 'Event/px', 'Event/py', 'Event/pz', 'Event/openingAngle', 'Event/conversion']

    with File(path2+os.path.basename(files1[i]), 'w') as f2:
        f2.create_dataset('ECAL', data=ecal_new)
        for feature in keepFeatures:
            f2.create_dataset(feature, data=f1[feature][:])
