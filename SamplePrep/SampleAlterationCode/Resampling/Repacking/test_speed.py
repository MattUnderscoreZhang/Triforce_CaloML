import h5py as h5
import sys

path = sys.argv[1]
data = h5.File(path)
for i, _ in enumerate(data['HCAL']):
    print(i)
