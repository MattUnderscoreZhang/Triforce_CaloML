import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import h5py as h5
import numpy as np

ele_path = "/data/LCD/NewSamples/RandomAngle/EleEscan_RandomAngle_MERGED/*.h5"
chpi_path = "/data/LCD/NewSamples/RandomAngle/ChPiEscan_RandomAngle_Filtered_1_Combined/*.h5"

###########
# COMBINE #
###########

ele_files = glob.glob(ele_path)
chpi_files = glob.glob(chpi_path)

features = [('ECAL_E', 600, 10), ('ECAL_nHits', 600, 10), ('ECAL_ratioFirstLayerToSecondLayerE', 3, 0.1), ('ECAL_ratioFirstLayerToTotalE', 0.01, 0.0005), ('ECALmomentX1', 100, 2), ('ECALmomentX2', 10, 0.2), ('ECALmomentX3', 10, 0.2), ('ECALmomentX4', 10, 0.2), ('ECALmomentX5', 10, 0.2), ('ECALmomentX6', 10, 0.2), ('ECALmomentY1', 100, 2), ('ECALmomentY2', 10, 0.2), ('ECALmomentY3', 10, 0.2), ('ECALmomentY4', 10, 0.2), ('ECALmomentY5', 10, 0.2), ('ECALmomentY6', 10, 0.2), ('ECALmomentZ1', 100, 2), ('ECALmomentZ2', 100, 2), ('ECALmomentZ3', 10, 0.2), ('ECALmomentZ4', 10, 0.2), ('ECALmomentZ5', 10, 0.2), ('ECALmomentZ6', 10, 0.2), ('HCAL_E', 10, 0.2), ('HCAL_ECAL_ERatio', 10, 0.2), ('HCAL_ECAL_nHitsRatio', 10, 0.2), ('HCAL_nHits', 100, 2), ('HCAL_ratioFirstLayerToSecondLayerE', 10, 0.2), ('HCAL_ratioFirstLayerToTotalE', 10, 0.2), ('HCALmomentX1', 10, 0.2), ('HCALmomentX2', 10, 0.2), ('HCALmomentX3', 10, 0.2), ('HCALmomentX4', 10, 0.2), ('HCALmomentX5', 10, 0.2), ('HCALmomentX6', 10, 0.2), ('HCALmomentY1', 10, 0.2), ('HCALmomentY2', 10, 0.2), ('HCALmomentY3', 10, 0.2), ('HCALmomentY4', 10, 0.2), ('HCALmomentY5', 10, 0.2), ('HCALmomentY6', 10, 0.2), ('HCALmomentZ1', 10, 0.2), ('HCALmomentZ2', 10, 0.2), ('HCALmomentZ3', 10, 0.2), ('HCALmomentZ4', 10, 0.2), ('HCALmomentZ5', 10, 0.2), ('HCALmomentZ6', 10, 0.2), ('energy', 1000, 10), ('eta', 1.5, 0.05), ('theta', 10, 0.2)]

combined_ele_data = {}
combined_chpi_data = {}

def add_data(file_name, particle_type):
    file = h5.File(file_name, 'r')
    if particle_type == 0:
        combined_data = combined_ele_data
    elif particle_type == 1:
        combined_data = combined_chpi_data
    for (feature, _, _) in features:
        if feature not in combined_data.keys():
            combined_data[feature] = list(file[feature][:])
        else:
            combined_data[feature] += list(file[feature][:])
    file.close()

for file_name in ele_files:
    add_data(file_name, 0)
for file_name in chpi_files:
    add_data(file_name, 1)

########
# PLOT #
########

for (feature, max_bin, bin_step) in features:
    bins=np.arange(0, max_bin, bin_step)
    if feature in ['eta', 'theta']:
        bins=np.arange(-max_bin, max_bin, bin_step)
    plt.hist(np.clip(combined_ele_data[feature], bins[0], bins[-1]), bins=bins, normed=True, histtype='step', linewidth=1, label='Ele')
    plt.hist(np.clip(combined_chpi_data[feature], bins[0], bins[-1]), bins=bins, normed=True, histtype='step', linewidth=1, label='ChPi')
    plt.title(feature + ' with Overflow Bin')
    plt.xlabel(feature)
    plt.ylabel('Normalized Fraction')
    plt.legend()
    plt.savefig("Plots/" + feature + '_ratios.png')
    plt.clf()
