import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from triforce_helper_functions import eval
import numpy as np
import sklearn as sk

class Analyzer():

    #############
    # HISTORIES #
    #############
    def plot_loss_vs_epoch(train, test, filename): 

        plt.figure()
        plt.title("Classifier Loss History")
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.plot(range(train), train, 'o-', color="r", label="Training Loss")
        plt.plot(range(train), test, 'o-', color="g", label="Test Loss")
        plt.legend(loc="best")
        plt.save(filename)

    def plot_accuracy_vs_epoch(train, test, filename): 

        plt.figure()
        plt.title("Classifier Accuracy History")
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.plot(range(train), train, 'o-', color="r", label="Training Accuracy")
        plt.plot(range(train), test, 'o-', color="g", label="Test Accuracy")
        plt.legend(loc="best")
        plt.save(filename)

    def plot_loss_vs_batches(train, test, filename):

        plt.figure()
        plt.title("Classifier Loss History")
        plt.xlabel("Training Time")
        plt.ylabel("Loss")
        plt.grid()
        plt.plot(range(train), train, 'o-', color="r", label="Training Loss")
        plt.plot(range(train), test, 'o-', color="g", label="Test Loss")
        plt.legend(loc="best")
        plt.save(filename)

    def plot_accuracy_vs_batches(train, test, filename):

        plt.figure()
        plt.title("Classifier Accuracy History")
        plt.xlabel("Training Time")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.plot(range(train), train, 'o-', color="r", label="Training Accuracy")
        plt.plot(range(train), test, 'o-', color="g", label="Test Accuracy")
        plt.legend(loc="best")
        plt.save(filename)

    #############
    # ROC CURVE #
    #############

    def plot_ROC(outputs, truth, filename):

        fpr, tpr, thresholds = sk.metrics.roc_curve(truth, outputs)
        roc_auc = sk.metrics.auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Classification')
        plt.legend(loc="lower right")
        plt.save(filename)

    ##########################
    # MAIN ANALYSIS FUNCTION #
    ##########################

    def analyze(self, tools, testLoader, out_file):

        [classifier, regressor, GAN] = tools

        classifier_test_results = []
        for data in testLoader:
            ECALs, HCALs, ys, energies = data
            ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
            if (classifier != None): classifier_test_results.append(eval(classifier, ECALs, HCALs, ys))
            else: classifier_test_results.append((0,0,0,0,0,0))

        print(classifier_test_results)
        classifier_test_results = np.array(classifier_test_results)
        n_test_batches = classifier_test_results.shape[0]
        classifier_test_loss = classifier_test_results[:,0].sum() / n_test_batches
        classifier_test_accuracy = classifier_test_results[:,1].sum() / n_test_batches
        classifier_test_outputs = classifier_test_results[:,2]
        classifier_test_truth = classifier_test_results[:,3]

        print('test loss: (C) %.4f; test accuracy: (C) %.4f' % (classifier_test_loss, classifier_test_accuracy))
        if (classifier != None): out_file.create_dataset("classifier_test_results", data=classifier_test_results)

        folder = out_file.filename[:out_file.filename.rfind('/')]
        plot_loss_vs_batches(out_file['classifier_loss_history_train'], out_file['classifier_loss_history_test'], folder+"/loss_history_batches.png")
        plot_accuracy_vs_batches(out_file['classifier_accuracy_history_train'], out_file['classifier_accuracy_history_test'], folder+"/accuracy_history_batches.png")
        plot_loss_vs_epoch(out_file['classifier_loss_epoch_train'], out_file['classifier_loss_epoch_test'], folder+"/loss_epoch_batches.png")
        plot_accuracy_vs_epoch(out_file['classifier_accuracy_epoch_train'], out_file['classifier_accuracy_epoch_test'], folder+"/accuracy_epoch_batches.png")
        plot_ROC(classifier_test_outputs, classifier_test_truth, folder+"/ROC.png")
