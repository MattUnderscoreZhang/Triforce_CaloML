import h5py as h5
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from matplotlib.patches import Rectangle

GAN = h5.File("/data/LCD/GAN/V0/EleEscan/EleContinous.hdf5")
Geant = h5.File("/data/LCD/V1/EleEscan/EleEscan_1_1.h5")

GAN_ECAL = GAN['ECAL'][()]/100
Geant_ECAL = Geant['ECAL'][()]
GAN_energy = GAN['target'][()].flatten() # use this to bin by energy
Geant_energy = Geant['target'][:,1].flatten()

energyLow = 100
energyHigh = 200
GAN_ECAL_bin = GAN_ECAL[np.logical_and(GAN_energy>=energyLow, GAN_energy<energyHigh)]
Geant_ECAL_bin = Geant_ECAL[np.logical_and(Geant_energy>=energyLow, Geant_energy<energyHigh)]

GAN_ECAL_bin = GAN_ECAL_bin[:1000]
Geant_ECAL_bin = Geant_ECAL_bin[:1000]
GAN_ECAL_bin = GAN_ECAL_bin.flatten()
Geant_ECAL_bin = Geant_ECAL_bin.flatten()

GAN_ECAL_bin = GAN_ECAL_bin[GAN_ECAL_bin > 0]
Geant_ECAL_bin = Geant_ECAL_bin[Geant_ECAL_bin > 0]

# maxRange = 0.05
maxRange = 0.005
GAN_ECAL_bin = GAN_ECAL_bin[GAN_ECAL_bin < maxRange]
Geant_ECAL_bin = Geant_ECAL_bin[Geant_ECAL_bin < maxRange]

handles = [Rectangle((0,0), 1, 1, edgecolor=c, fill=False) for c in ['b', 'g']]
labels = ['GAN', 'Geant']

bins = np.histogram(np.hstack((GAN_ECAL_bin, Geant_ECAL_bin)), bins=40)[1]
GAN_plot = plt.hist(GAN_ECAL_bin, bins=bins, histtype='step', color='b', normed=False)
Geant_plot = plt.hist(Geant_ECAL_bin, bins=bins, histtype='step', color='g', normed=False)
plt.legend(handles=handles, labels=labels)
plt.title("ECAL Cell Energies")
ax = plt.gca()
# ax.set_ylim([0,ax.get_ylim()[1]*1.5])
ax.set_yscale('log')
plt.show()
