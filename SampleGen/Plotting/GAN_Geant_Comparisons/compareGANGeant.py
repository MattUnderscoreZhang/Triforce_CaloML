import h5py as h5
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from matplotlib.patches import Rectangle

# GAN = h5.File("Data/GANEle_0to500GeV.h5")
# Geant = h5.File("Data/GeantEle_0to500GeV.h5")
GAN = h5.File("Data/GANPiPlus_0to500GeV.h5")
Geant = h5.File("Data/GeantPiPlus_0to500GeV.h5")

def compare(title, key):

    print key
    GAN_key = GAN[key][()]
    Geant_key = Geant[key][()]
    GAN_energy = GAN['Energy'][()] # use this to bin by energy
    Geant_energy = Geant['Energy'][()]

    GAN_energy = GAN_energy[np.isfinite(GAN_key)] # take only finite values
    Geant_energy = Geant_energy[np.isfinite(Geant_key)]
    GAN_key = GAN_key[np.isfinite(GAN_key)]
    Geant_key = Geant_key[np.isfinite(Geant_key)]

    # maxRange = 20
    # GAN_energy = GAN_energy[GAN_key<maxRange]
    # Geant_energy = Geant_energy[Geant_key<maxRange]
    # GAN_key = GAN_key[GAN_key<maxRange]
    # Geant_key = Geant_key[Geant_key<maxRange]

    energyBins = [100, 200, 300, 400, 500]
    # energyBins = [400, 420, 440, 460, 480, 500]
    # energyBins = [100, 500]

    for i in range(len(energyBins)-1):

        energyLow = energyBins[i]
        energyHigh = energyBins[i+1]

        GAN_key_bin = GAN_key[np.logical_and(GAN_energy>=energyLow, GAN_energy<energyHigh)]
        Geant_key_bin = Geant_key[np.logical_and(Geant_energy>=energyLow, Geant_energy<energyHigh)]
        # GAN_key_bin = GAN_key_bin/(100) # correct GAN ECAL_E

        handles = [Rectangle((0,0), 1, 1, edgecolor=c, fill=False) for c in ['b', 'g']]
        labels = ['GAN', 'Geant']

        bins = np.histogram(np.hstack((GAN_key_bin, Geant_key_bin)), bins=40)[1]
        GAN_plot = plt.hist(GAN_key_bin, bins=bins, histtype='step', color='b', normed=True)
        Geant_plot = plt.hist(Geant_key_bin, bins=bins, histtype='step', color='g', linestyle='--', normed=True)
        plt.legend(handles=handles, labels=labels)
        plt.xlabel(title)
        ax = plt.gca()
        # ax.set_yscale('log')
        ax.set_ylim([0,ax.get_ylim()[1]*1.5])
        plt.savefig('Plots/PiPlus/'+str(energyLow)+"to"+str(energyHigh)+"/"+title+'.png')
        plt.cla(); plt.clf()

features = []
def h5_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if prefix=='': path = key
        if isinstance(item, h5.Dataset):
            yield path
        elif isinstance(item, h5.Group):
            for data in h5_dataset_iterator(item, path): yield data
for path in h5_dataset_iterator(GAN):
    features.append(path)

for feature in features:
    if 'HCAL' not in feature:
        compare(feature.split('/')[-1], feature)
# compare('ECAL_E', 'ECAL/ECAL_E')
# compare('ECALmomentX5', 'ECAL_Moments/ECALmomentX5')
# compare('ECAL_nHist', 'ECAL/ECAL_nHits')
