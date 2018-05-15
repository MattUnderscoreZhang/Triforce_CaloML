# figuring out what the best opening angle cut should be for Gamma and Pi0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

gamma_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/GammaEscan_*_MERGED/GammaEscan_*.h5"
pi0_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/Pi0Escan_*_MERGED/Pi0Escan_*.h5"

###########
# COMBINE #
###########

gamma_files = glob.glob(gamma_path)
pi0_files = glob.glob(pi0_path)

gamma_opening_angle = []
gamma_E = []
pi0_opening_angle = []
pi0_E = []

for file_name in gamma_files:
    file = h5.File(file_name, 'r')
    gamma_opening_angle += list(file['openingAngle'][:])
    gamma_E += list(file['energy'][:])
    file.close()

for file_name in pi0_files:
    file = h5.File(file_name, 'r')
    pi0_opening_angle += list(file['openingAngle'][:])
    pi0_E += list(file['energy'][:])
    file.close()

bins=np.arange(0,10,0.1)
plt.hist(np.clip(gamma_opening_angle, bins[0], bins[-1]), bins=bins, density=True, histtype='step', label='Gamma')
plt.hist(np.clip(pi0_opening_angle, bins[0], bins[-1]), bins=bins, density=True, histtype='step', label='Pi0')
plt.xlabel('Opening Angle')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.savefig('opening_angles.png')

plt.clf()

bins=np.arange(0,1,0.01)
plt.hist(np.clip(gamma_opening_angle, bins[0], bins[-1]), bins=bins, density=True, histtype='step', label='Gamma')
plt.hist(np.clip(pi0_opening_angle, bins[0], bins[-1]), bins=bins, density=True, histtype='step', label='Pi0')
plt.xlabel('Opening Angle')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.savefig('zoom_opening_angles.png')

plt.clf()

bin_gamma_E_means = binned_statistic(gamma_opening_angle, gamma_E, bins=50, range=(0, 5)).statistic
bin_pi0_E_means = binned_statistic(pi0_opening_angle, pi0_E, bins=50, range=(0, 5)).statistic

plt.plot(np.arange(0,5,0.1), bin_gamma_E_means, label='Gamma')
plt.plot(np.arange(0,5,0.1), bin_pi0_E_means, label='Pi0')
plt.title('Mean Energy in Opening Angle Bins')
plt.xlabel('Opening Angle')
plt.ylabel('Energy')
plt.legend()
plt.savefig('opening_angle_vs_energy.png')

plt.clf()

zip(*[(r, E) for r, E in zip(pi0_opening_angle, pi0_E) if r < 0.1])

plt.plot(np.arange(0,5,0.1), bin_gamma_E_means, label='Gamma')
plt.plot(np.arange(0,5,0.1), bin_pi0_E_means, label='Pi0')
plt.title('Mean Energy in Opening Angle Bins')
plt.xlabel('Opening Angle')
plt.ylabel('Energy')
plt.legend()
plt.savefig('opening_angle_vs_energy.png')
