# Make plots from hyperparameter scans

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itr
import pdb

# Options
# files = glob.glob("/data/LCD/NewSamples/HyperparameterResults/DNN/*.txt")
# net_type = "DNN"
# parameter_names = ["Hidden Layers", "Neurons Per Hidden Layer", "Learning Rate", "Dropout Probability"]
# parameter_space = [[1, 2, 3, 4, 5], [128, 256, 512, 1024, 2048], [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007], [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]]
# parameter_shape = [len(dimension) for dimension in parameter_space]
# baseline_parameters = (2, 2, 3, 3)

# files = glob.glob("/data/LCD/NewSamples/HyperparameterResults/CNN/*.txt")
# net_type = "CNN"
# parameter_names = ["Hidden Layers", "Neurons Per Hidden Layer", "Learning Rate", "Dropout Probability", "Number of ECAL Filters", "Number of ECAL Kernels"]
# parameter_space = [[1, 2, 3, 4, 5], [128, 256, 512, 1024, 2048], [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007], [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
# parameter_shape = [len(dimension) for dimension in parameter_space]
# baseline_parameters = (2, 2, 3, 3, 2, 2)

files = glob.glob("/data/LCD/NewSamples/HyperparameterResults/GN/*.txt")
net_type = "GN"
parameter_names = ["Neurons in Final Hidden Layer", "Learning Rate", "Decay Rate"]
parameter_space = [[128, 256, 512, 1024, 2048, 4096], [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007], [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]]
parameter_shape = [len(dimension) for dimension in parameter_space]
baseline_parameters = (2, 3, 3)

# Extract losses and accuracies
losses = np.empty(shape=(parameter_shape), dtype=object)
losses.fill(-1)
accuracies = np.empty(shape=(parameter_shape), dtype=object)
accuracies.fill(-1)
for file_name in files:
    # get classifier test loss and accuracy
    with open(file_name) as current_file:
        lines = current_file.readlines()
    # results = lines[-3].rstrip().replace(';', '')
    results = lines[-3].rstrip().replace(' ',  '')
    results = results.replace(":", ";").split(";")
    classifier_indices = [i.replace('.','',1).isdigit() for i in results]
    results = [float(i) for (i,j) in zip(results, classifier_indices) if j]
    if len(results) == 0:
        continue
    test_loss = results[0]
    test_accuracy = results[1]
    # parse filename
    file_name = file_name.split("/")[-1]
    file_name = file_name.split(".log")[0]
    parameters = file_name.split("_")[2:-2]
    parameters = [float(i) for i in parameters]
    split = int(file_name.split("_")[-2][4:])
    # store values
    p = [parameter_space[i].index(x) for i, x in enumerate(parameters)]
    if losses[tuple(p)] == -1:
        losses[tuple(p)] = [test_loss]
        accuracies[tuple(p)] = [test_accuracy]
    else:
        losses[tuple(p)].append(test_loss)
        accuracies[tuple(p)].append(test_accuracy)

# # Plot 1D scans
# for parameter_i in range(len(parameter_space)):
    # losses_1D = np.empty(parameter_shape[parameter_i])
    # accuracy_1D = np.empty(parameter_shape[parameter_i])
    # index = baseline_parameters
    # for i in range(parameter_shape[parameter_i]):
        # index = list(index)
        # index[parameter_i] = i
        # index = tuple(index)
        # losses_1D[i] = losses[index]
        # accuracy_1D[i] = accuracies[index]
    # plt.plot(parameter_space[parameter_i],accuracy_1D, 'b+-', linewidth = 1, markersize=18)
    # plt.xlabel(parameter_names[parameter_i])
    # plt.ylabel("accuracy")
    # plt.title("Accuracy vs " + parameter_names[parameter_i])
    # plt.show()
    # plt.plot(parameter_space[parameter_i],losses_1D, 'r+-', linewidth = 1, markersize = 18)
    # plt.xlabel(parameter_names[parameter_i])
    # plt.ylabel("loss")
    # plt.title("Loss vs " + parameter_names[parameter_i])
    # plt.show()

# Plot 2D scans
combinations = list(itr.combinations(range(len(parameter_shape)), 2))
for k in combinations:
    x = k[0]
    y = k[1]
    losses_mean = np.empty([parameter_shape[x],parameter_shape[y]])
    losses_std = np.empty([parameter_shape[x],parameter_shape[y]])
    losses_label = np.empty([parameter_shape[x],parameter_shape[y]], dtype=object)
    accuracy_mean = np.empty([parameter_shape[x],parameter_shape[y]])
    accuracy_std = np.empty([parameter_shape[x],parameter_shape[y]])
    accuracy_label = np.empty([parameter_shape[x],parameter_shape[y]], dtype=object)
    for i in range(parameter_shape[x]):
        for j in range(parameter_shape[y]):
            index = list(baseline_parameters)
            index[x] = i
            index[y] = j
            index = tuple(index)
            losses_mean[i][j] = np.mean(losses[index])
            losses_std[i][j] = np.std(losses[index])
            losses_label[i][j] = str(losses_mean[i][j])[:6] + "\n+/-\n" + str(losses_std[i][j])[:6]
            accuracy_mean[i][j] = np.mean(accuracies[index])
            accuracy_std[i][j] = np.std(accuracies[index])
            accuracy_label[i][j] = str(accuracy_mean[i][j])[:6] + "\n+/-\n" + str(accuracy_std[i][j])[:6]
    acc = pd.DataFrame(accuracy_mean, index = parameter_space[x], columns = parameter_space[y])
    loss = pd.DataFrame(losses_mean, index = parameter_space[x], columns = parameter_space[y])
    plt.clf()
    ax = sns.heatmap(acc, annot = accuracy_label, fmt = "", annot_kws={"size": 7})
    plt.xlabel(parameter_names[y])
    plt.ylabel(parameter_names[x])
    plt.title("Accuracy")
    plt.savefig("Plots/HyperparameterScan/"+net_type+"/"+str(x)+"_"+str(y)+".eps")
    # ax = sns.heatmap(loss, annot = True, fmt = "")
    # plt.xlabel(parameter_names[y])
    # plt.ylabel(parameter_names[x])
    # plt.title("Loss")
    # plt.show()
