import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

import h5py
import numpy as np

def scatter3d(x,y,z, values, colorsMap='jet'):

    cm = plt.get_cmap(colorsMap) # set up color map
    cNorm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values)) # normalize to min and max of data
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm) # map data using color scale

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(values)) # plot 3D grid
    ax.set_title("Energy in calorimeter")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    scalarMap.set_array(values)
    fig.colorbar(scalarMap) # plot color bar
    plt.show()

data = h5py.File("../AllFiles/SkimmedH5Files/gamma_60_GeV_1.h5")

ECAL = data['ECAL']
x, y, z = np.nonzero(ECAL[0]) # first event
E = ECAL[0][np.nonzero(ECAL[0])]
print x.shape, y.shape, z.shape, E.shape
scatter3d(x,y,z, E)

HCAL = data['HCAL']
x, y, z = np.nonzero(HCAL[0]) # first event
E = HCAL[0][np.nonzero(HCAL[0])]
print x.shape, y.shape, z.shape, E.shape
scatter3d(x,y,z, E)
