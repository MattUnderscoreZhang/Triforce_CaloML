import h5py as h5
import numpy as np

def plot_ECAL(ECAL, save_name):
    x,y,z = ECAL.nonzero() 
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(x, y, -z, marker='.', zdir='z', c=ECAL[x,y,z], cmap='jet', alpha=0.3) 
    ax.set_xlabel('X') 
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z') 
    plt.savefig(save_name)

def spoof_ATLAS_geometry(ECAL):
    # geometry conversion
    new_ECAL = np.empty_like(ECAL)
    x, y, z = ECAL.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                pass
    return new_ECAL

def spoof_CMS_geometry(ECAL):
    x, y, z = ECAL.shape
    new_ECAL = ECAL.sum(2).shape # collapse in z direction
    new_ECAL = #interpolate
    new_ECAL = np.tile(new_ECAL/z, (z,1,1))
    return new_ECAL

data = h5.File("/data/LCD/NewSamples/RandomAngle/EleEscan_RandomAngle_MERGED/EleEscan_RandomAngle_1_1.h5")
ECAL = data['ECAL']

for i, ECAL_event in enumerate(ECAL):
    if i > 10:
        break
    plot_ECAL(ECAL_event, "Plots/ECAL_"+str(i)+"_before.png"
    new_ECAL = spoof_CMS_geometry(ECAL_event)
    plot_ECAL(new_ECAL, "Plots/ECAL_"+str(i)+"_after.png"
