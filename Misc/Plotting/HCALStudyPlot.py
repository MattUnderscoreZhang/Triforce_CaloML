import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

# [particles][HCAL][stat]
# particles - 0 = Ele/ChPi; 1 = Gamma/Pi0
# HCAL - 0 = without; 1 = with
# stat - 0 = train accuracy; 1 = train loss; 2 = test accuracy; 3 = test loss
stat_name = ["Training Accuracy", "Training Loss", "Test Accuracy", "Test Loss"]

ele_chpi_paths = ["/home/mazhang/Triforce_CaloML/Output/NIPS_EleChPi_NoHCAL_*/training_results.h5", "/home/mazhang/Triforce_CaloML/Output/NIPS_EleChPi_WithHCAL_*/training_results.h5"]
gamma_pi0_paths = ["/home/mazhang/Triforce_CaloML/Output/NIPS_GammaPi0_NoHCAL_*/training_results.h5", "/home/mazhang/Triforce_CaloML/Output/NIPS_GammaPi0_WithHCAL_*/training_results.h5"]
paths = [ele_chpi_paths, gamma_pi0_paths]

data = [[[[] for i in range(4)] for j in range(2)] for k in range(2)]
data_error = [[[[] for i in range(4)] for j in range(2)] for k in range(2)]
for particles in range(2):
    for HCAL in range(2):
        files = glob.glob(paths[particles][HCAL])
        for file_name in files:
            file = h5.File(file_name, 'r')
            data[particles][HCAL][0].append(list(file['accuracy_classifier_train_batch'][:30]))
            data[particles][HCAL][1].append(list(file['loss_classifier_train_batch'][:30]))
            data[particles][HCAL][2].append(list(file['accuracy_classifier_test_batch'][:30]))
            data[particles][HCAL][3].append(list(file['loss_classifier_test_batch'][:30]))
            file.close()

# sns.set_style("whitegrid")
for particles in range(2):
    for HCAL in range(2):
        for stat in range(4):
            data_error[particles][HCAL][stat] = np.sqrt(np.mean(np.square(data[particles][HCAL][stat]), axis=0))/len(data[particles][HCAL][stat]) # approximation of std using n instead of (n-1)
            data[particles][HCAL][stat] = np.mean(data[particles][HCAL][stat], axis=0)
            x = np.arange(1, len(data[particles][HCAL][0])+1)
            plt.clf()
            plt.errorbar(x, data[particles][HCAL][stat], data_error[particles][HCAL][stat], fmt='o-', markersize=8)
            plt.xlabel("Batch Number")
            plt.xlabel(stat_name[stat])
            plt.title(stat_name[stat])
            xmax = max(x) + 5
            ymax = max([i+j for (i, j) in zip(data[particles][HCAL][0], data_error[particles][HCAL][0])]) + 0.3
            plt.xlim(0, xmax)
            plt.ylim(0, ymax)
            plt.savefig(str(particles)+"_"+str(HCAL)+"_"+str(stat)+".png")
