# figuring out what the best H/E energy ratio cut should be for Ele and ChPi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

files = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/RandomAngle/EleEscan_*_MERGED/EleEscan_*.h5"
keys = ['energy', 'eta', 'phi']

files = glob.glob(files)
data = {}
for key in keys:
    data[key] = []

###########
# COMBINE #
###########

for file_name in files:
    file = h5.File(file_name, 'r')
    data[key] += list(file[key][:])
    file.close()

bins=np.arange(0,10,0.2)
plt.hist(np.clip(data['energy'], bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1)
plt.title('Energy')
plt.xlabel('Energy')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.savefig('energy.png')
plt.clf()

bins=np.arange(0,10,0.2)
plt.hist(np.clip(data['eta'], bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1)
plt.title('Eta')
plt.xlabel('Eta')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.savefig('eta.png')
plt.clf()

bins=np.arange(0,10,0.2)
plt.hist(np.clip(data['phi'], bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1)
plt.title('Phi')
plt.xlabel('Phi')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.savefig('phi.png')
plt.clf()
