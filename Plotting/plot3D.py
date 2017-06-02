import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import os

import h5py
import numpy as np

#####################
# PLOTTING FUNCTION #
#####################

def scatter3d(x, y, z, values, colorsMap='jet', saveName='Plots/Test.pdf'):

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

    #plt.show()
    plt.savefig(saveName, bbox_inches="tight")

#########################
# CHOOSE EVENTS TO PLOT #
#########################

data = h5py.File("../AllFiles/H5Files/v1/Skimmed/Pi0/pi0_60_GeV_1.h5")
events = [0, 1, 2, 3, 4]

saveDirectory = "Plots/"
if not os.path.exists(saveDirectory): os.makedirs(saveDirectory)

# plot each event
for eventN in events:

    ECAL = data['ECAL']
    x, y, z = np.nonzero(ECAL[eventN]) # first event
    E = ECAL[eventN][np.nonzero(ECAL[eventN])]
    scatter3d(x, y, z, E, saveName=saveDirectory+"ECAL_event"+str(eventN)+".pdf")

    HCAL = data['HCAL']
    x, y, z = np.nonzero(HCAL[eventN]) # first event
    E = HCAL[eventN][np.nonzero(HCAL[eventN])]
    scatter3d(x, y , z, E, saveName=saveDirectory+"HCAL_event"+str(eventN)+".pdf")

# plot averages
ECAL_average = np.average(data['ECAL'], axis=0)
x, y, z = np.nonzero(ECAL_average) # first event
E = ECAL_average[np.nonzero(ECAL_average)]
scatter3d(x, y, z, E, saveName=saveDirectory+"ECAL_average.pdf")

HCAL_average = np.average(data['HCAL'], axis=0)
x, y, z = np.nonzero(HCAL_average) # first event
E = HCAL_average[np.nonzero(HCAL_average)]
scatter3d(x, y , z, E, saveName=saveDirectory+"HCAL_average.pdf")
