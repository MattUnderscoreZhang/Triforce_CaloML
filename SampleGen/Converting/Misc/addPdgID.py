import sys
import glob
import h5py as h5
import numpy as np

path = sys.argv[1]
pdgID = int(sys.argv[2])

files = glob.glob(path)

for file in files:
    print "Adding pdgID", pdgID, "to file", file
    data = h5.File(file)
    data.create_dataset("pdgID", data=np.array([pdgID]*10000))
