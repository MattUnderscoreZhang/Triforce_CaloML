import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

CLIC = h5.File("/public/data/calo/RandomAngle/CLIC2/Gamma/GammaEscan_RandomAngle_1_1.h5")
CMS = h5.File("/public/data/calo/RandomAngle/CMS/Gamma/GammaEscan_RandomAngle_1_1.h5")
ATLAS = h5.File("/public/data/calo/RandomAngle/ATLAS/Gamma/GammaEscan_RandomAngle_1_1.h5")
savePath = '../../Plots/'

CLIC_R9 = CLIC['R9'][()]
CMS_R9 = CMS['R9'][()]
ATLAS_R9 = ATLAS['R9'][()]

CLIC_R9 = CLIC_R9[np.isfinite(CLIC_R9)]
CMS_R9 = CMS_R9[np.isfinite(CMS_R9)]
ATLAS_R9 = ATLAS_R9[np.isfinite(ATLAS_R9)]

plt.hist(CLIC_R9, bins=np.arange(0, 1, 0.01), histtype='step', color='b', label='CLIC', normed=True)
plt.hist(CMS_R9, bins=np.arange(0, 1, 0.01), histtype='step', color='r', label='CMS', normed=True)
plt.hist(ATLAS_R9, bins=np.arange(0, 1, 0.01), histtype='step', color='g', label='ATLAS', normed=True)
plt.legend()

plt.title("R9 for Different Detector Geometries")
ax = plt.gca()
# ax.set_yscale('log')
ax.set_ylim([0, ax.get_ylim()[1]*1.5])

plt.savefig(savePath + "/GeometryComparisonR9.eps")
