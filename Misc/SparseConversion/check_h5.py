import h5py as h5
import numpy as np
import timeit

start = timeit.default_timer()
ecal = np.zeros((10000, 51, 51, 25))
data = h5.File("sparse_data.h5")['ECAL']
for event in data:
    for i in range(len(event)/5):
        ecal[tuple([int(i) for i in event[i*5:i*5+4]])] = event[i*5+4]
end = timeit.default_timer()
print(end-start)
