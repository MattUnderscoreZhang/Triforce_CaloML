# Make plots from hyperparameter scans

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itr
import pdb

# Files
files = glob.glob("Data/*.txt")

# Options
parameter_names = ["Hidden Layers", "Neurons per Hidden Layer", "Learning Rate", "Dropout Probability"]
parameter_space = [[1, 2, 3, 4, 5], [256, 512, 1024, 2048, 4096], [0.0001, 0.0005, 0.001], [0.05, 0.1, 0.15, 0.2, 0.25]]
parameter_shape = [len(dimension) for dimension in parameter_space]
baseline_parameters = (1, 2, 1, 1)

# Extract losses and accuracies
losses = np.zeros(parameter_shape)
accuracies = np.zeros(parameter_shape)
for file_name in files:
    # get classifier test loss and accuracy
    with open(file_name) as current_file:
        lines = current_file.readlines()
    results = lines[-2].rstrip().replace(';', '')
    results = results.split(" ")
    classifier_indices = [i for i, x in enumerate(results) if x == '(C)']
    test_loss = results[classifier_indices[0]+1]
    test_accuracy = results[classifier_indices[1]+1]
    # parse filename
    file_name = file_name.split("/")[-1]
    file_name = file_name.split(".log")[0]
    parameters = file_name.split("_")[1:-1]
    parameters = [float(i) for i in parameters]
    # store values
    p = [parameter_space[i].index(x) for i, x in enumerate(parameters)]
    losses[p[0]][p[1]][p[2]][p[3]][p[4]] = test_loss
    accuracies[p[0]][p[1]][p[2]][p[3]][p[4]] = test_accuracy

# Plot 1D scans
for parameter_i in range(len(parameter_space)):
    losses_1D = np.empty(parameter_shape[parameter_i])
    accuracy_1D = np.empty(parameter_shape[parameter_i])
    index = baseline_parameters
    for i in range(parameter_shape[parameter_i]):
        index = list(index)
        index[parameter_i] = i
        index = tuple(index)
        losses_1D[i] = losses[index]
        accuracy_1D[i] = accuracies[index]
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
combinations = list(itr.combinations(range(5), 2))
for k in combinations:
    x = k[0]
    y = k[1]
    lossesx = np.empty([parameter_shape[x],parameter_shape[y]])
    accuracyx = np.empty([parameter_shape[x],parameter_shape[y]])
    for i in range(parameter_shape[x]):
        for j in range(parameter_shape[y]):
            index = list(baseline_parameters)
            index[x] = i
            index[y] = j
            index = tuple(index)
            lossesx[i][j] = losses[index]
            accuracyx[i][j] = accuracies[index]
    acc = pd.DataFrame(accuracyx, index = parameter_space[x], columns = parameter_space[y])
    loss = pd.DataFrame(lossesx, index = parameter_space[x], columns = parameter_space[y])
    plt.clf()
    ax = sns.heatmap(acc, annot = True, fmt = "")
    plt.xlabel(parameter_names[y])
    plt.ylabel(parameter_names[x])
    plt.title("Accuracy")
    plt.savefig(str(x)+"_"+str(y)+".pdf")
    # ax = sns.heatmap(loss, annot = True, fmt = "")
    # plt.xlabel(parameter_names[y])
    # plt.ylabel(parameter_names[x])
    # plt.title("Loss")
    # plt.show()
