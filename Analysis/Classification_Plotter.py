import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from statistics import mean
import pdb
from scipy.stats import binned_statistic

class Analyzer():

    ####################
    # TRAINING HISTORY #
    ####################

    def plot_history(self, train, test, loss, batch, filename): 

        plt.figure()
        title = "Classifier "
        if loss:
            title += "Loss "
            plt.ylabel("Loss")
        else:
            title += "Accuracy "
            plt.ylabel("Accuracy")
        if batch:
            title += "History"
            plt.xlabel("Training Batch")
        else:
            title += "vs Epoch"
            plt.xlabel("Training Epoch")
        plt.title(title)
        plt.grid('on', linestyle='--')
        if loss:
            plt.plot(range(1, train.shape[0]+1), train, 'o-', color="g", label="Training Loss", alpha=0.5)
            plt.plot(range(1, test.shape[0]+1), test, 'o-', color="r", label="Test Loss", alpha=0.5)
        else:
            plt.plot(range(1, train.shape[0]+1), train, 'o-', color="g", label="Training Accuracy", alpha=0.5)
            plt.plot(range(1, test.shape[0]+1), test, 'o-', color="r", label="Test Accuracy", alpha=0.5)
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

    ###################
    # BINNED ACCURACY #
    ###################

    def plot_accuracy_bins(self, bin_feature, final_val_results, filename):
        class_acc = (final_val_results['class_prediction'] == final_val_results['class_truth']).cpu().numpy()
        class_feature = np.array(final_val_results[bin_feature]).flatten()
        n_bins = 50
        if bin_feature == 'energy':
            n_bins = 49
            bin_range = (10, 500)
        elif bin_feature == 'eta':
            bin_range = (-5, 5)
        elif bin_feature == 'opening_angle':
            bin_range = (-5, 5)
        bin_class_acc = binned_statistic(class_feature, class_acc, bins=n_bins, range=bin_range).statistic
        plt.plot(np.arange(bin_range[0],bin_range[1],(bin_range[1]-bin_range[0])/n_bins), bin_class_acc)
        plt.title('Mean classification accuracy in ' + bin_feature + ' bins')
        plt.xlabel(bin_feature)
        plt.ylabel('accuracy')
        plt.grid()
        plt.savefig(filename)
        plt.clf()

    ##########################
    # MAIN ANALYSIS FUNCTION #
    ##########################

    def analyze(self, tools, test_train_history, final_val_results):

        [combined_classifier, discriminator, generator] = tools

        classifier_test_loss = mean(final_val_results['class_reg_loss'])
        classifier_test_accuracy = mean(final_val_results['class_acc'])
        classifier_test_scores = final_val_results['class_prediction']
        classifier_test_truth = final_val_results['class_truth']

        print('test loss: %8.4f; test accuracy: %8.4f' % (classifier_test_loss, classifier_test_accuracy))
        test_train_history.create_dataset("classifier_test_accuracy", data=classifier_test_accuracy) 

        folder = test_train_history.filename[:test_train_history.filename.rfind('/')]

        self.plot_accuracy_bins('energy', final_val_results, folder+"/accuracy_vs_energy.png")
        self.plot_accuracy_bins('eta', final_val_results, folder+"/accuracy_vs_eta.png")
        self.plot_accuracy_bins('opening_angle', final_val_results, folder+"/accuracy_vs_opening_angle.png")

        self.plot_history(test_train_history['class_reg_loss_train_batch'], test_train_history['class_reg_loss_test_batch'], loss=True, batch=True, filename=folder+"/loss_batches.png")
        self.plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename=folder+"/accuracy_batches.png")
        self.plot_history(test_train_history['class_reg_loss_train_batch'], test_train_history['class_reg_loss_test_batch'], loss=True, batch=True, filename=folder+"/loss_batches.png")
        self.plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename=folder+"/loss_batches.png")
