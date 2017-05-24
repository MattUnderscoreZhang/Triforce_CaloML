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
    sizes = values/max(values)*50

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, s=sizes, c=scalarMap.to_rgba(values)) # plot 3D grid
    ax.set_title("Energy in calorimeter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Layer")

    scalarMap.set_array(values)
    fig.colorbar(scalarMap) # plot color bar
    plt.show()

data = h5py.File("../AllFiles/SkimmedH5Files/pi0_60_GeV_1.h5")
events = [0, 1, 2, 3, 4]

for eventN in events:

    ECAL = data['ECAL']
    x, y, z = np.nonzero(ECAL[eventN]) # first event
    E = ECAL[eventN][np.nonzero(ECAL[eventN])]
    print x.shape, y.shape, z.shape, E.shape
    scatter3d(x,y,z, E)

    HCAL = data['HCAL']
    x, y, z = np.nonzero(HCAL[eventN]) # first event
    E = HCAL[eventN][np.nonzero(HCAL[eventN])]
    print x.shape, y.shape, z.shape, E.shape
    scatter3d(x,y,z, E)
