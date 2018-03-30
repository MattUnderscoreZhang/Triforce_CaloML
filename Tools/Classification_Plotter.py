# Plotting acc along epoch and ROC curve stuff

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, h5py
import pdb

def accuracy_vs_batches(results, interval): 
    """
    results = 'Output/Output008/results.h5'
    Parameters
    @results: the paht of output file.h5 storing training and test results
    @interval: the interval between each sample point
    """

    f = h5py.File(results, 'r')

    data = f['classifier_accuracy_history_train']

    if not os.path.exists(os.path.join(results.split('/')[0], 'Plots')): 
        os.makedirs(os.path.join(results.split('/')[0], 'Plots'))
    output_path = os.path.join(results.split('/')[0], 'Plots', 'accuracy_vs_batches.png')

    fig, ax = plt.subplots()
    sample_points = [(i + 1) * interval for i in range(len(data))]
    # plt.style.use('ggplot')
    ax.plot(sample_points, data, color='b')
    ax.grid()
    ax.set_xlabel('Batches of training')
    ax.set_ylabel('Accuracy')
    ax.set_title('Trends of Accuracy along Batches')

    plt.savefig(output_path)

def accuracy_vs_epochs(results): 
    """
    results = 'Output/Output008/results.h5'
    Parameters
    @results: the paht of output file.h5 storing training and test results
    """

    f = h5py.File(results, 'r')

    data = f['classifier_accuracy_epoch_train']

    if not os.path.exists(os.path.join(results.split('/')[0], 'Plots')): 
        os.makedirs(os.path.join(results.split('/')[0], 'Plots'))
    output_path = os.path.join(results.split('/')[0], 'Plots', 'accuracy_vs_epochs.png')

    fig, ax = plt.subplots()
    sample_points = [(i + 1) for i in range(len(data))]
    ax.plot(sample_points, data, 'ro', sample_points, data, 'b')
    ax.grid()
    ax.set_xlabel('Epochs of training')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.set_ylabel('Accuracy')
    ax.set_title('Trends of Accuracy along Epochs')

    plt.savefig(output_path)

def loss_vs_batches(results, interval): 
    """
    """
    f = h5py.File(results, 'r')
    data = f['classifier_loss_history_train']

    if not os.path.exists(os.path.join(results.split('/')[0], 'Plots')): 
        os.makedirs(os.path.join(results.split('/')[0], 'Plots'))
    output_path = os.path.join(results.split('/')[0], 'Plots', 'loss_vs_batches.png')

    fig, ax = plt.subplots()
    sample_points = [(i + 1) * interval for i in range(len(data))]
    # plt.style.use('ggplot')
    ax.plot(sample_points, data, color='b')
    ax.grid()
    ax.set_xlabel('Batches of training')
    ax.set_ylabel('Loss')
    ax.set_title('Trends of Loss along Batches')

    plt.savefig(output_path)

def loss_vs_epochs(results): 
    """
    results = 'Output/Output008/results.h5'
    Parameters
    @results: the paht of output file.h5 storing training and test results
    """

    f = h5py.File(results, 'r')

    data = f['classifier_loss_epoch_train']

    if not os.path.exists(os.path.join(results.split('/')[0], 'Plots')): 
        os.makedirs(os.path.join(results.split('/')[0], 'Plots'))
    output_path = os.path.join(results.split('/')[0], 'Plots', 'loss_vs_epochs.png')

    fig, ax = plt.subplots()
    sample_points = [(i + 1) for i in range(len(data))]
    ax.plot(sample_points, data, 'ro', sample_points, data, 'b')
    ax.grid()
    ax.set_xlabel('Epochs of training')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.set_ylabel('Loss')
    ax.set_title('Trends of Loss along Epochs')

    plt.savefig(output_path)

def make_all(results, accuracy_interval=20, loss_interval=20): 
    """
    Making all possible plots for classification
    """
    accuracy_vs_batches(f, accuracy_interval)
    accuracy_vs_epochs(f)
    loss_vs_batches(f, loss_interval)
    loss_vs_epochs(f)

if __name__ == '__main__': 
    f = '../results.h5'
    make_all(f)