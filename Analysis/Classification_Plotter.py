import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from statistics import mean
import pdb

class Analyzer():

    #############
    # HISTORIES #
    #############
    def plot_loss_vs_epoch(self, train, test, filename): 

        plt.figure()
        plt.title("Classifier Loss vs Epoch")
        plt.xlabel("Training Epoch")
        plt.ylabel("Loss")
        plt.grid('on', linestyle='--')
        plt.plot(range(1, train.shape[0]+1), train, 'o-', color="g", label="Training Loss", alpha=0.5)
        plt.plot(range(1, test.shape[0]+1), test, 'o-', color="r", label="Test Loss", alpha=0.5)
        plt.legend(loc="best")
        plt.savefig(filename)

    def plot_accuracy_vs_epoch(self, train, test, filename): 

        plt.figure()
        plt.title("Classifier Accuracy vs Epoch")
        plt.xlabel("Training Epoch")
        plt.ylabel("Accuracy")
        plt.grid('on', linestyle='--')
        plt.plot(range(1, train.shape[0]+1), train, 'o-', color="g", label="Training Accuracy", alpha=0.5)
        plt.plot(range(1, test.shape[0]+1), test, 'o-', color="r", label="Test Accuracy", alpha=0.5)
        plt.ylim(ymax=1.0)
        plt.legend(loc="best")
        plt.savefig(filename)

    def plot_loss_vs_batches(self, train, test, filename):

        plt.figure()
        plt.title("Classifier Loss History")
        plt.xlabel("Training Batches")
        plt.ylabel("Loss")
        plt.grid('on', linestyle='--')
        plt.plot(range(train.shape[0]), train, 'o-', color="g", label="Training Loss", alpha=0.5)
        plt.plot(range(test.shape[0]), test, 'o-', color="r", label="Test Loss", alpha=0.5)
        plt.legend(loc="best")
        plt.savefig(filename)

    def plot_accuracy_vs_batches(self, train, test, filename):

        plt.figure()
        plt.title("Classifier Accuracy History")
        plt.xlabel("Training Batches")
        plt.ylabel("Accuracy")
        plt.grid('on', linestyle='--')
        plt.plot(range(train.shape[0]), train, 'o-', color="g", label="Training Accuracy", alpha=0.5)
        plt.plot(range(test.shape[0]), test, 'o-', color="r", label="Test Accuracy", alpha=0.5)
        plt.ylim(ymax=1.0)
        plt.legend(loc="best")
        plt.savefig(filename)

    #############
    # ROC CURVE #
    #############

    def plot_ROC(self, scores, truth, filename):

        fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid('on', linestyle='--')
        plt.title('ROC Curve for Classification')
        plt.legend(loc="lower right")
        plt.savefig(filename)

    ##########################
    # MAIN ANALYSIS FUNCTION #
    ##########################

    def analyze(self, tools, classifier_test_results, out_file):

        [combined_classifier, discriminator, generator] = tools

        classifier_test_loss = mean(classifier_test_results['class_reg_loss'])
        classifier_test_accuracy = mean(classifier_test_results['class_acc'])
        pdb.set_trace()
        classifier_test_scores = classifier_test_results['class_prediction']
        pdb.set_trace()
        classifier_test_truth = classifier_test_results['class_truth']
        pdb.set_trace()

        print('test loss: %8.4f; test accuracy: %8.4f' % (classifier_test_loss, classifier_test_accuracy))
        out_file.create_dataset("classifier_test_accuracy", data=classifier_test_accuracy) 

        folder = out_file.filename[:out_file.filename.rfind('/')]
        self.plot_loss_vs_batches(out_file['loss_classifier_train_batch'].value, out_file['loss_classifier_test_batch'].value, folder+"/loss_batches.png")
        self.plot_accuracy_vs_batches(out_file['accuracy_classifier_train_batch'].value, out_file['accuracy_classifier_test_batch'].value, folder+"/accuracy_batches.png")
        self.plot_loss_vs_epoch(out_file['loss_classifier_train_epoch'].value, out_file['loss_classifier_test_epoch'].value, folder+"/loss_epoch_batches.png")
        self.plot_accuracy_vs_epoch(out_file['accuracy_classifier_train_epoch'].value, out_file['accuracy_classifier_test_epoch'].value, folder+"/accuracy_epoch_batches.png")
        self.plot_ROC(classifier_test_scores, classifier_test_truth, folder+"/ROC.png")
