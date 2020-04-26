import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pathlib


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


if __name__ == "__main__":
    '''python plotECAL.py <file_name> <out_folder> <n_events> <ECAL/HCAL>'''
    file_name = sys.argv[1]
    out_folder = sys.argv[2]
    n_events = int(sys.argv[3])
    cal_type = sys.argv[4] # ECAL or HCAL
    assert(cal_type in ["ECAL", "HCAL"])

    pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

    ECAL = h5.File(file_name)[cal_type]
    for i in range(n_events):
        plot_ECAL(ECAL[i], pathlib.Path(out_folder)/(cal_type+"_"+str(i)+".eps"))
