import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

"""Figuring out what the best opening angle cut should be for Gamma and Pi0."""

# pi0_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/Pi0Escan_*_MERGED/Pi0Escan_*.h5"
pi0_path = "/public/data/Calo/RandomAngle/CLIC/Pi0/Pi0Escan*.h5"

###########
# COMBINE #
###########

pi0_files = glob.glob(pi0_path)

pi0_opening_angle = []
pi0_E = []

for file_name in pi0_files:
    file = h5.File(file_name, 'r')
    pi0_opening_angle += list(file['openingAngle'][:])
    pi0_E += list(file['energy'][:])
    file.close()

# sns.set(font_scale=1.2)

bins = np.arange(0, 0.1, 0.002)
plt.hist(np.clip(pi0_opening_angle, bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1, label='Pi0')
# plt.title('Opening Angle of Pi0 Events with Overflow Bin', fontsize=15)
plt.xlabel('Opening Angle (Radians)', fontsize=12)
plt.ylabel('Normalized Fraction', fontsize=12)
plt.yscale('log')
plt.savefig('opening_angles.png')

plt.clf()

bins = np.arange(0, 0.01, 0.0005)
fig, ax = plt.subplots()
ax.hist(np.clip(pi0_opening_angle, bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1, label='Pi0')
# plt.title('Opening Angle of Pi0 Events with Overflow Bin', fontsize=15)
ax.set_xlabel('Opening Angle (Radians)', fontsize=12)
ax.set_ylabel('Normalized Fraction', fontsize=12)
ax.set_yscale('log')

ax2 = ax.twiny()
ax2.set_xlim(0, 0.01/0.003)
ax2.set_xlabel('n ECAL cells')
plt.savefig('zoom_opening_angles.png')

plt.clf()

bin_pi0_E_means = binned_statistic(pi0_opening_angle, pi0_E, bins=20, range=(0, 0.01)).statistic

plt.plot(np.arange(0, 0.01, 0.0005), bin_pi0_E_means, label='Pi0')
# plt.title('Mean Pi0 Energy in Opening Angle Bins', fontsize=15)
plt.xlabel('Opening Angle (Radians)', fontsize=12)
plt.ylabel('Energy (GeV)', fontsize=12)
plt.ylim(0, 450)
# plt.grid()
plt.savefig('mean_energy_vs_opening_angle.png')

plt.clf()

pi0_pass_cut = [a < 0.01 for a in pi0_opening_angle]
bin_pi0_pass_cut_means = binned_statistic(pi0_E, pi0_pass_cut, bins=50, range=(0, 500)).statistic

plt.plot(np.arange(0, 500, 10), bin_pi0_pass_cut_means, label='Pi0')
# plt.title('Fraction of Pi0 Events Passing Opening Angle < 0.01 Cut', fontsize=15)
plt.xlabel('Energy (GeV)', fontsize=12)
plt.ylabel('Fraction', fontsize=12)
plt.savefig('opening_angle_cut_vs_energy.png')
