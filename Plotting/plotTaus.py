import h5py as h5
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
gamma = h5.File("GammaEscan/GammaEscan_0.h5")
pi0 = h5.File("Pi0Escan/Pi0Escan_0.h5")
gamma_tau1 = gamma['features'][:,-5]
gamma_tau2 = gamma['features'][:,-4]
gamma_tau2_over_tau1 = gamma['features'][:,-3]
gamma_tau3 = gamma['features'][:,-2]
gamma_tau3_over_tau2 = gamma['features'][:,-1]
pi0_tau1 = pi0['features'][:,-5]
pi0_tau2 = pi0['features'][:,-4]
pi0_tau2_over_tau1 = pi0['features'][:,-3]
pi0_tau3 = pi0['features'][:,-2]
pi0_tau3_over_tau2 = pi0['features'][:,-1]
plt.hist([gamma_tau3_over_tau2, pi0_tau3_over_tau2], color=['g', 'r'], histtype='step', bins=np.arange(0,0.5,0.01), linewidth=1.5)
plt.xlabel("tanh(tau3_over_tau2)")
labels= ["gamma", "pi0"]
plt.legend(labels)
plt.show()
plt.scatter(gamma_tau1, gamma_tau2, color='g', alpha=0.1)
plt.scatter(pi0_tau1, pi0_tau2, color='r', alpha=0.1)
plt.xlim((0, 0.01))
plt.ylim((0, 0.01))
plt.show()
