import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pathlib


def plot_ECAL(ECAL, save_name):
    cm = plt.get_cmap('jet')  # set up color map
    cNorm = matplotlib.colors.Normalize(vmin=min(ECAL), vmax=max(ECAL))  # normalize to min and max of data
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cm)  # map data using color scale
    sizes = pow(ECAL*100, 2)/max(ECAL)

    x, y, z = ECAL.nonzero()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, marker='.', zdir='z', c=scalarMap.to_rgba(ECAL), cmap='jet', alpha=0.3, s=sizes)

    ax.set_title("Energy in calorimeter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Layer")
    scalarMap.set_array(ECAL)
    fig.colorbar(scalarMap)  # plot color bar

    plt.savefig(save_name)


if __name__ == "__main__":
    '''python plotECAL.py <file_name> <out_folder> <n_events>'''
    file_name = sys.argv[1]
    out_folder = sys.argv[2]
    n_events = int(sys.argv[3])

    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    ECAL = h5.File(file_name)['ECAL']
    for i in range(n_events):
        plot_ECAL(ECAL[i], pathlib.Path(out_folder)/("ECAL_"+str(i)+".eps"))
