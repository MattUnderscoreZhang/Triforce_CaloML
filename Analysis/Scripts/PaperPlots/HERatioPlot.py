import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

"""Figuring out what the best H/E energy ratio cut should be for Ele and ChPi."""

# ele_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/EleEscan_*_MERGED/EleEscan_*.h5"
# chpi_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/ChPiEscan_*_MERGED/ChPiEscan_*.h5"
ele_path = "/public/data/Calo/RandomAngle/CLIC/Ele/EleEscan*.h5"
chpi_path = "/public/data/Calo/RandomAngle/CLIC/ChPi/ChPiEscan*.h5"

###########
# COMBINE #
###########

ele_files = glob.glob(ele_path)
chpi_files = glob.glob(chpi_path)

ele_ratio = []
ele_E = []
chpi_ratio = []
chpi_E = []

for file_name in ele_files:
    file = h5.File(file_name, 'r')
    ele_ratio += list(file['HCAL_ECAL_ERatio'][:])
    ele_E += list(file['energy'][:])
    file.close()

for file_name in chpi_files:
    file = h5.File(file_name, 'r')
    chpi_ratio += list(file['HCAL_ECAL_ERatio'][:])
    chpi_E += list(file['energy'][:])
    file.close()

# sns.set(font_scale=1.2)

bins = np.arange(0, 10, 0.2)
plt.hist(np.clip(ele_ratio, bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1, label='Ele')
plt.hist(np.clip(chpi_ratio, bins[0], bins[-1]), bins=bins, density=True, histtype='step', linewidth=1, label='ChPi')
# plt.title('H/E Ratio of Electron and ChPi Events with Overflow Bin', fontsize=15)
plt.xlabel('HCAL_ECAL_ERatio', fontsize=12)
plt.ylabel('Normalized Fraction', fontsize=12)
plt.yscale('log')
plt.legend()
plt.savefig('ratios.png')

plt.clf()

bins = np.arange(0, 1, 0.05)
plt.hist(ele_ratio, bins=bins, density=True, histtype='step', linewidth=1, label='Ele')
plt.hist(chpi_ratio, bins=bins, density=True, histtype='step', linewidth=1, label='ChPi')
# plt.title('H/E Ratio of Electron and ChPi Events', fontsize=15)
plt.xlabel('HCAL_ECAL_ERatio', fontsize=12)
plt.ylabel('Normalized Fraction', fontsize=12)
plt.yscale('log')
plt.legend()
plt.savefig('zoom_ratios.png')

plt.clf()

# bin_chpi_E_means = binned_statistic(chpi_ratio, chpi_E, bins=20, range=(0, 5)).statistic

# plt.plot(np.arange(0, 5, 0.25), bin_chpi_E_means, label='ChPi')
# # plt.title('Mean ChPi Energy in H/E Energy Ratio Bins', fontsize=15)
# plt.xlabel('HCAL_ECAL_ERatio', fontsize=12)
# plt.ylabel('Energy (GeV)', fontsize=12)
# # plt.grid()
# plt.savefig('mean_energy_vs_ratio.png')

# plt.clf()

nbins = 20
cut_events = [(i, j) for (i, j) in zip(chpi_ratio, chpi_E) if i < 5]
cut_chpi_ratio = [i[0] for i in cut_events]
cut_chpi_E = [i[1] for i in cut_events]
nevents, _ = np.histogram(cut_chpi_ratio, bins=nbins)
bin_yields, _ = np.histogram(cut_chpi_ratio, bins=nbins, weights=cut_chpi_E)
bin_sq_yields, _ = np.histogram(cut_chpi_ratio, bins=nbins, weights=[i*i for i in cut_chpi_E])
mean = bin_yields / nevents
std = np.sqrt(bin_sq_yields/nevents - mean*mean)/np.sqrt(nevents)

# plt.plot(np.arange(0, 5, 0.25), bin_chpi_E_means, label='ChPi')
# plt.title('Mean ChPi Energy in H/E Energy Ratio Bins', fontsize=15)
# plt.plot(cut_chpi_ratio, cut_chpi_E, 'bo')
plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
plt.xlabel('HCAL_ECAL_ERatio', fontsize=12)
plt.ylabel('Energy (GeV)', fontsize=12)
# plt.grid()
plt.savefig('mean_energy_vs_ratio.png')

plt.clf()

ele_pass_cut = [r < 0.1 for r in ele_ratio]
chpi_pass_cut = [r < 0.1 for r in chpi_ratio]
bin_ele_pass_cut_means = binned_statistic(ele_E, ele_pass_cut, bins=50, range=(0, 500)).statistic
bin_chpi_pass_cut_means = binned_statistic(chpi_E, chpi_pass_cut, bins=50, range=(0, 500)).statistic

plt.plot(np.arange(0, 500, 10), bin_ele_pass_cut_means, label='Ele')
plt.plot(np.arange(0, 500, 10), bin_chpi_pass_cut_means, label='ChPi')
# plt.title('Fraction of Events Passing H/E < 0.1 Cut', fontsize=15)
plt.xlabel('Energy (GeV)', fontsize=12)
plt.ylabel('Fraction', fontsize=12)
plt.legend()
plt.savefig('ratio_cut_vs_energy.png')
