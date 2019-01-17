import h5py as h5
import numpy as np
import timeit

start = timeit.default_timer()
for n in range(1, 11):
    ecal = h5.File("/data/LCD/NewSamples/Fixed/EleEscan_1_MERGED/EleEscan_1_" + str(n) + ".h5")['ECAL'][:]
end = timeit.default_timer()
print("Time to load 10 normal data files:", end-start)

start = timeit.default_timer()
for n in range(1, 11):
    ecal = np.zeros((10000, 51, 51, 25))
    data = h5.File("/data/LCD/NewSamples/Fixed/Sparse/EleEscan_1_" + str(n) + ".h5")['ECAL']
    for event in data:
        for i in range(len(event)/5):
            ecal[tuple([int(i) for i in event[i*5:i*5+4]])] = event[i*5+4]
end = timeit.default_timer()
print("Time to load 10 sparse data files:", end-start)
