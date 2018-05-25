# Make plots from hyperparameter scans

import glob
import numpy as np

files = "/home/mazhang/Triforce_CaloML/Misc/Plotting/Logs/*"
files = glob.glob(files)

parameter_space = [[4, 5, 6], [256, 512, 1024], [0.001, 0.005, 0.01]]
parameter_shape = [len(dimension) for dimension in parameter_space]
n_parameters = len(parameter_shape)
losses = np.empty(parameter_shape)
accuracies = np.empty(parameter_shape)

for file_name in files:
    # get classifier test loss and accuracy
    file = open(file_name)
    lines = file.readlines()
    file.close()
    results = lines[-2].rstrip().replace(';', '')
    results = results.split(" ")
    indices = [i for i, x in enumerate(results) if x == '(C)']
    test_loss = results[indices[0]+1]
    test_accuracy = results[indices[1]+1]
    # parse filename
    file_name = file_name.split("/")[-1]
    file_name = file_name.split(".log")[0]
    parameters = file_name.split("_")[1:]
    parameters = [float(i) for i in parameters]
    # store values
    p = [parameter_space[i].index(x) for i, x in enumerate(parameters)]
    losses[p[0]][p[1]][p[2]] = test_loss
    accuracies[p[0]][p[1]][p[2]] = test_accuracy

print losses
