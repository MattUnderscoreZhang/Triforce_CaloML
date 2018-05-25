import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

# [particles][HCAL][stat][split]
particles_name = ["Ele/ChPi", "Gamma/Pi0"]
# HCAL - 0 = without; 1 = with
stat_name = ["Accuracy", "Loss"]
split_name = ["Training", "Test"]

ele_chpi_paths = ["/home/mazhang/Triforce_CaloML/Output/NIPS_EleChPi_NoHCAL_*/training_results.h5", "/home/mazhang/Triforce_CaloML/Output/NIPS_EleChPi_WithHCAL_*/training_results.h5"]
gamma_pi0_paths = ["/home/mazhang/Triforce_CaloML/Output/NIPS_GammaPi0_NoHCAL_*/training_results.h5", "/home/mazhang/Triforce_CaloML/Output/NIPS_GammaPi0_WithHCAL_*/training_results.h5"]
paths = [ele_chpi_paths, gamma_pi0_paths]

data = [[[[[] for h in range(2)] for i in range(2)] for j in range(2)] for k in range(2)]
data_error = [[[[[] for h in range(2)] for i in range(2)] for j in range(2)] for k in range(2)]
for particles in range(2):
    for HCAL in range(2):
        files = glob.glob(paths[particles][HCAL])
        for file_name in files:
            file = h5.File(file_name, 'r')
            data[particles][HCAL][0][0].append(list(file['accuracy_classifier_train_batch'][:30]))
            data[particles][HCAL][0][1].append(list(file['accuracy_classifier_test_batch'][:30]))
            data[particles][HCAL][1][0].append(list(file['loss_classifier_train_batch'][:30]))
            data[particles][HCAL][1][1].append(list(file['loss_classifier_test_batch'][:30]))
            file.close()
        for stat in range(2):
            for split in range(2):
                data_error[particles][HCAL][stat][split] = np.sqrt(np.mean(np.square(data[particles][HCAL][stat][split]), axis=0))/len(data[particles][HCAL][stat][split]) # approximation of std using n instead of (n-1)
                data[particles][HCAL][stat][split] = np.mean(data[particles][HCAL][stat][split], axis=0)

# sns.set_style("whitegrid")
for particles in range(2):
    for stat in range(2):
        x = np.arange(1, len(data[particles][0][stat][0])+1)
        plt.clf()
        plt.errorbar(x, data[particles][0][stat][0], data_error[particles][0][stat][0], fmt='o-', markersize=3, label='Training, No HCAL')
        plt.errorbar(x, data[particles][1][stat][0], data_error[particles][1][stat][0], fmt='o-', markersize=3, label='Training, HCAL')
        plt.errorbar(x, data[particles][0][stat][1], data_error[particles][0][stat][1], fmt='o-', markersize=3, label='Test, No HCAL')
        plt.errorbar(x, data[particles][1][stat][1], data_error[particles][1][stat][1], fmt='o-', markersize=3, label='Training, HCAL')
        plt.xlabel("Batch Number")
        plt.ylabel("Average " + stat_name[stat] + " Over 10 Trials")
        plt.title(particles_name[particles] + " " + stat_name[stat])
        if stat == 0:
            plt.legend(loc = 'lower right')
        elif stat == 1:
            plt.legend(loc = 'upper right')
        xmax = max(x) + 5
        ymax_train_noHCAL = max([i+j for (i, j) in zip(data[particles][0][stat][0], data_error[particles][0][stat][1])])
        ymax_train_HCAL = max([i+j for (i, j) in zip(data[particles][1][stat][0], data_error[particles][1][stat][1])])
        ymax_test_noHCAL = max([i+j for (i, j) in zip(data[particles][0][stat][0], data_error[particles][0][stat][1])])
        ymax_test_HCAL = max([i+j for (i, j) in zip(data[particles][1][stat][0], data_error[particles][1][stat][1])])
        ymax_train = max(ymax_train_noHCAL, ymax_train_HCAL)
        ymax_test = max(ymax_test_noHCAL, ymax_test_HCAL)
        ymax = max(ymax_train, ymax_test) + 0.3
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)
        # plt.yscale('log')
        plt.savefig(str(particles)+"_"+str(stat)+".png")
