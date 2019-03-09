import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


data = h5.File("/home/matt/Projects/calo/FinalResults/GAN/Ele_GAN_GEANT_results.h5")
GAN_ID = 0
GEANT_ID = 1
save_path = 'Plots/'


if __name__ == "__main__":

    GAN_indices = np.array(data['pdgID']) == GAN_ID
    GEANT_indices = np.array(data['pdgID']) == GEANT_ID

    GAN_energy = data['energy'][GAN_indices]
    GEANT_energy = data['energy'][GEANT_indices]

    GAN_reg_energy_prediction = data['reg_energy_prediction'][GAN_indices]
    GEANT_reg_energy_prediction = data['reg_energy_prediction'][GEANT_indices]

    rcParams['axes.titlepad'] = 20
    plt.scatter(GAN_energy, GAN_reg_energy_prediction, s=1, label='GAN')
    plt.scatter(GEANT_energy, GEANT_reg_energy_prediction, s=1, label='GEANT')
    plt.xlim([90, 210])
    plt.ylim([50, 250])
    plt.title("Energy Predictions from Regression Nets for GAN and GEANT4 Samples")
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.legend()
    plt.savefig(save_path + "GAN_GEANT_energy_regression_comparison.eps")

    GAN_class_prediction = data['class_prediction'][GAN_indices]
    GEANT_class_prediction = data['class_prediction'][GEANT_indices]

    GAN_class_prediction_accuracy = sum(GAN_class_prediction) / len(GAN_class_prediction)
    GEANT_class_prediction_accuracy = sum(GEANT_class_prediction) / len(GEANT_class_prediction)

    plt.clf()
    plt.bar(range(2), [GAN_class_prediction_accuracy, GEANT_class_prediction_accuracy], align='center', alpha=0.5)
    plt.xticks(range(2), ["GAN", "GEANT"])
    plt.ylim([0.95, 1.01])
    plt.title("Accuracy of Classification Nets on GAN and GEANT4 Electron Samples")
    plt.savefig(save_path + "GAN_GEANT_class_accuracy_comparison.eps")
