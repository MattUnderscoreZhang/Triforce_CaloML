import h5py as h5
import numpy as np

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
    # save plot of original ECAL
    new_ECAL = spoof_CMS_geometry(ECAL_event)
    # save plot of new ECAL 
