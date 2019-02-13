import h5py as h5
import sys

path = sys.argv[1]
data = h5.File(path)
for i, _ in enumerate(data['ECAL']):
    print(i)
    if i > 1000:
        break
for i, _ in enumerate(data['HCAL']):
    print(i)
    if i > 1000:
        break
