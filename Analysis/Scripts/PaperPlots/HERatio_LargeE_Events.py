import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import h5py as h5
import numpy as np

"""Look at events that have large energy but H/E==0."""

# ele_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/EleEscan_*_MERGED/EleEscan_*.h5"
# chpi_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/ChPiEscan_*_MERGED/ChPiEscan_*.h5"
ele_path = "/public/data/Calo/RandomAngle/CLIC/Ele/EleEscan*.h5"
chpi_path = "/public/data/Calo/RandomAngle/CLIC/ChPi/ChPiEscan*.h5"

ele_files = glob.glob(ele_path)
chpi_files = glob.glob(chpi_path)


def plot_ECAL(ECAL, save_name):
    x, y, z = ECAL.nonzero()
    values = ECAL[ECAL.nonzero()].flatten()

    cm = plt.get_cmap('jet')  # set up color map
    cNorm = matplotlib.colors.Normalize(vmin=values.min(), vmax=values.max())  # normalize to min and max of data
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)  # map data using color scale
    sizes = pow(values*100, 1.3)/values.max()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker='.', zdir='z', c=scalarMap.to_rgba(values), cmap='jet', alpha=0.8, s=sizes)

    # ax.set_title("Energy in calorimeter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Layer")
    scalarMap.set_array(ECAL)
    # fig.colorbar(scalarMap)  # plot color bar

    plt.savefig(save_name)


for file_name in ele_files + chpi_files:
    file = h5.File(file_name, 'r')
    HE_ratios = np.array(file['HCAL_ECAL_ERatio'][:])
    energies = np.array(file['energy'][:])
    indices = np.intersect1d(np.where(HE_ratios == 0), np.where(energies > 400))
    for i in indices:
        # plot_ECAL(file["ECAL"][i], "blah.png")
        print(sum(sum(sum(file["ECAL"][i]))))
    file.close()
