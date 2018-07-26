import h5py as h5
import numpy as np
import pdb

data = h5.File("/data/LCD/NewSamples/RandomAngle/ChPiEscan_RandomAngle_MERGED/ChPiEscan_RandomAngle_1_10.h5")
ecal = data['ECAL']
print("Starting conversion")
indices = np.transpose(np.nonzero(ecal))
indices = [list(i) for i in indices]
print("Number of non-zero values:", len(indices))

out_file = h5.File('sparse_data.h5')
dtype = h5.special_dtype(vlen=np.dtype('float'))
dset = out_file.create_dataset('ECAL', (len(ecal),), dtype=dtype)

for i, index in enumerate(indices):
    if (i%10000==0): print(i, "out of", len(indices))
    a = np.append(dset[index[0]], index+[ecal[tuple(index)]])
    dset[index[0]] = np.append(dset[index[0]], index+[ecal[tuple(index)]])
