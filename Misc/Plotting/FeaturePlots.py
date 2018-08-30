# figuring out what the best H/E energy ratio cut should be for Ele and ChPi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

files = "/data/LCD/NewSamples/RandomAngle/EleEscan_*_MERGED/EleEscan_*.h5"
keys = ['energy', 'eta', 'phi']
ranges = [(0,510,5), (-1,1,0.05), (0,0.5,0.01)]

files = glob.glob(files)
data = {}
for key in keys:
    data[key] = []

###########
# COMBINE #
###########

for file_name in files:
    file = h5.File(file_name, 'r')
    for key in keys:
        data[key] += list(file[key][:])
    file.close()

for key, range in zip(keys, ranges):
    bins = np.arange(range[0],range[1],range[2])
    # weights = np.ones_like(data[key])/float(len(data[key]))
    # plt.hist(np.clip(data[key], bins[0], bins[-1]), bins=bins, weights=weights, histtype='step', linewidth=1)
    plt.hist(np.clip(data[key], bins[0], bins[-1]), bins=bins, normed=True, histtype='step', linewidth=1)
    plt.title(key.capitalize())
    plt.xlabel(key)
    plt.ylabel('Normalized Fraction')
    plt.savefig(key+'.png')
    plt.clf()
