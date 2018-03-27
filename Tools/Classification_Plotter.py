# Plotting acc along epoch and ROC curve stuff

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, h5py
import pdb

def accuracy_vs_events(results, interval): 
    """
    results = h5py.File('Output/Output008/results.h5')
    Parameters
    @results: the output file.h5 storing training and test results
    @interval: the interval between each sample point
    """

    data = results['classifier_accuracy_history_train']

    fig, ax = plt.subplots()
    sample_points = [(i + 1) * interval for i in range(len(data))]
    # plt.style.use('ggplot')
    ax.plot(sample_points, data, color='b')
    ax.grid()
    ax.set_xlabel('Epoch of training')
    ax.set_ylabel('Accuracy')
    ax.set_title('Trends of Accuracy along Epochs')

    plt.savefig('test.png')

    # pdb.set_trace()

if __name__ == '__main__': 
    f = h5py.File('../results.h5', 'r')
    accuracy_vs_events(f, 20)