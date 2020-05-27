import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

data = h5.File("/public/data/Calo/RandomAngle/CLIC/Gamma/GammaEscan_RandomAngle_1_1.h5")
R9 = data["R9"]
ECAL = data["ECAL"]

print(R9[:10])
for i in range(10):
    plt.imshow(np.sum(ECAL[i], 2))
    print(R9[i])
    plt.show()
