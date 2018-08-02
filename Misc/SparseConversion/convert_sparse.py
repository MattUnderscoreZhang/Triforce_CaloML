import h5py as h5
import numpy as np
import sys

def sparsify(in_file_name, out_file_name):

    data = h5.File(in_file_name)
    ecal = data['ECAL']
    print("Starting conversion")
    indices = np.transpose(np.nonzero(ecal))
    indices = [list(i) for i in indices]
    print("Number of non-zero values:", len(indices))

    out_file = h5.File(out_file_name)
    dtype = h5.special_dtype(vlen=np.dtype('float'))
    dset = out_file.create_dataset('ECAL', (len(ecal),), dtype=dtype)

    for i, index in enumerate(indices):
        if (i%10000==0): print(i, "out of", len(indices))
        a = np.append(dset[index[0]], index+[ecal[tuple(index)]])
        dset[index[0]] = np.append(dset[index[0]], index+[ecal[tuple(index)]])

    # indices = [index+[ecal[tuple(index)]] for index in indices]
    # pdb.set_trace()

if __name__ == "__main__":

    in_file_name = sys.argv[1]
    out_file_name = sys.argv[2]
    print("Sparsifying", in_file_name, "and storing to", out_file_name)
    sparsify(in_file_name, out_file_name)
