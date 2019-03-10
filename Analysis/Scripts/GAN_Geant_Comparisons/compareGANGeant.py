import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# GAN = h5.File("Data/GANEle_0to500GeV.h5")
# Geant = h5.File("Data/GeantEle_0to500GeV.h5")
GAN = h5.File("/public/data/calo/GAN/Ele_GAN.h5")
Geant = h5.File("/public/data/calo/GAN/Ele_GEANT4.h5")
savePath = 'Plots/'


def compare(title, key):

    print(key)
    GAN_key = GAN[key][()]
    Geant_key = Geant[key][()]
    GAN_energy = GAN['energy'][()]  # use this to bin by energy
    Geant_energy = Geant['energy'][()]

    GAN_energy = GAN_energy[np.isfinite(GAN_key)]  # take only finite values
    Geant_energy = Geant_energy[np.isfinite(Geant_key)]
    GAN_key = GAN_key[np.isfinite(GAN_key)]
    Geant_key = Geant_key[np.isfinite(Geant_key)]

    # maxRange = 20
    # GAN_energy = GAN_energy[GAN_key<maxRange]
    # Geant_energy = Geant_energy[Geant_key<maxRange]
    # GAN_key = GAN_key[GAN_key<maxRange]
    # Geant_key = Geant_key[Geant_key<maxRange]

    # energyBins = [100, 200, 300, 400, 500]
    # energyBins = [400, 420, 440, 460, 480, 500]
    energyBins = [0, 5000]

    x_limits = {'ECAL_ratioFirstLayerToSecondLayerE': (1, 10),
                'ECAL_ratioFirstLayerToTotalE': (0, 0.008),
                'ECALmomentX2': (0, 20),
                'ECALmomentX3': (-100, 300),
                'ECALmomentX4': (0, 8000),
                'ECALmomentX5': (-50000, 200000),
                'ECALmomentY1': (20, 30)}

    for i in range(len(energyBins)-1):

        energyLow = energyBins[i]
        energyHigh = energyBins[i+1]

        GAN_key_bin = GAN_key[np.logical_and(GAN_energy >= energyLow, GAN_energy < energyHigh)]
        Geant_key_bin = Geant_key[np.logical_and(Geant_energy >= energyLow, Geant_energy < energyHigh)]
        # GAN_key_bin = GAN_key_bin/(100) # correct GAN ECAL_E

        handles = [Rectangle((0, 0), 1, 1, edgecolor=c, fill=False) for c in ['b', 'g']]
        labels = ['GAN', 'Geant']

        if key in x_limits.keys():
            bins = np.histogram(np.hstack((GAN_key_bin, Geant_key_bin)), range=x_limits[key], bins=40)[1]
        else:
            bins = np.histogram(np.hstack((GAN_key_bin, Geant_key_bin)), bins=40)[1]
        plt.hist(GAN_key_bin, bins=bins, histtype='step', color='b', normed=True)
        plt.hist(Geant_key_bin, bins=bins, histtype='step', color='g', linestyle='--', normed=True)
        plt.legend(handles=handles, labels=labels)
        plt.xlabel(title)
        ax = plt.gca()
        # ax.set_yscale('log')
        ax.set_ylim([0, ax.get_ylim()[1]*1.5])
        plt.savefig(savePath+str(energyLow)+"to"+str(energyHigh)+"/"+title+'.png')
        plt.cla()
        plt.clf()


if __name__ == "__main__":

    def h5_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if prefix == '':
                path = key
            if isinstance(item, h5.Dataset):
                yield path
            elif isinstance(item, h5.Group):
                for data in h5_dataset_iterator(item, path):
                    yield data

    features = []
    for path in h5_dataset_iterator(GAN):
        features.append(path)

    for feature in features:
        if feature not in ['ECAL', 'HCAL']:
            compare(feature.split('/')[-1], feature)
    # compare('ECAL_E', 'ECAL/ECAL_E')
    # compare('ECALmomentX5', 'ECAL_Moments/ECALmomentX5')
    # compare('ECAL_nHist', 'ECAL/ECAL_nHits')
