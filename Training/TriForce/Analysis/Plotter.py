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
            if train.shape[0] > 100 or test.shape[0] > 100: 
                plt.plot(range(1, train.shape[0]+1), train, '-', color="g", label="Training Loss", alpha=0.5)
                plt.plot(range(1, test.shape[0]+1), test, '-', color="r", label="Test Loss", alpha=0.5)
            else: 
                plt.plot(range(1, train.shape[0]+1), train, 'o-', color="g", label="Training Loss", alpha=0.5)
                plt.plot(range(1, test.shape[0]+1), test, 'o-', color="r", label="Test Loss", alpha=0.5)
        else:
            plt.ylim(ymax=1.0)
            if train.shape[0] > 100 or test.shape[0] > 100: 
                plt.plot(range(1, train.shape[0]+1), train, '-', color="g", label="Training Accuracy", alpha=0.5)
                plt.plot(range(1, test.shape[0]+1), test, '-', color="r", label="Test Accuracy", alpha=0.5)
            else: 
                plt.plot(range(1, train.shape[0]+1), train, 'o-', color="g", label="Training Accuracy", alpha=0.5)
                plt.plot(range(1, test.shape[0]+1), test, 'o-', color="r", label="Test Accuracy", alpha=0.5)
        plt.legend(loc="best")
        plt.savefig(filename)
        plt.clf()

    #############
    # ROC CURVE #
    #############

    def plot_ROC(self, final_val_results, filename):

        truth = final_val_results['class_truth']
        scores = final_val_results['class_raw_prediction']
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
        plt.clf()

    ###################
    # BINNED ACCURACY #
    ###################

    def plot_accuracy_bins(self, bin_feature, final_val_results, filename):
        class_acc = (final_val_results['class_prediction'] == final_val_results['class_truth'])
        class_feature = np.array(final_val_results[bin_feature]).flatten()
        n_bins = 50
        if bin_feature == 'energy':
            n_bins = 49
            bin_range = (10, 500)
        elif bin_feature == 'eta':
            n_bins = 11
            bin_range = (-0.55, 0.55)
        elif bin_feature == 'openingAngle':
            bin_range = (0.0, 0.5)
        bin_class_acc = binned_statistic(class_feature, class_acc, bins=n_bins, range=bin_range).statistic
        plt.plot(np.arange(bin_range[0],bin_range[1],(bin_range[1]-bin_range[0])/n_bins), bin_class_acc)
        plt.title('Mean classification accuracy in ' + bin_feature + ' bins')
        plt.xlabel(bin_feature)
        plt.ylabel('accuracy')
        plt.grid()
        plt.savefig(filename)
        plt.clf()

    #############################
    # BINNED REGRESSION RESULTS #
    #############################

    def plot_regression_bins(self, bin_feature, plot_feature, final_val_results, filenames):
        true = np.asarray(final_val_results[plot_feature]).flatten()
        pred = final_val_results['reg_%s_prediction'%(plot_feature)].flatten()
        diff = true - pred
        if plot_feature == 'energy': diff = (diff/true) * 100.0
        binvar = np.asarray(final_val_results[bin_feature]).flatten()
        n_bins = 11
        if bin_feature == 'energy':
            n_bins = 25
            bin_range = (0, 500)
        elif bin_feature == 'eta':
            bin_range = (-0.55, 0.55)
        elif bin_feature == 'phi':
            bin_range = (0.0, 0.35)

        # plot mean
        bin_mean = binned_statistic(binvar, diff, statistic='mean', bins=n_bins, range=bin_range).statistic
        plt.plot(np.arange(bin_range[0],bin_range[1],(bin_range[1]-bin_range[0])/n_bins), bin_mean, marker='o')
        plt.title('Mean difference, ' + plot_feature + ' in bins of ' + bin_feature)
        plt.xlabel(bin_feature)
        plt.ylabel('Mean '+plot_feature)
        plt.grid(True,which='both')
        plt.savefig(filenames[0])
        plt.clf()

        # plot std dev
        bin_std = binned_statistic(binvar, diff, statistic=np.std, bins=n_bins, range=bin_range).statistic
        plt.plot(np.arange(bin_range[0],bin_range[1],(bin_range[1]-bin_range[0])/n_bins), bin_std, marker='o')
        plt.title('Stddev difference, ' + plot_feature + ' in bins of ' + bin_feature)
        plt.xlabel(bin_feature)
        plt.ylabel('Stddev '+plot_feature)
        if plot_feature == 'energy' and bin_feature == 'energy': plt.yscale('log')
        plt.grid(True,which='both')
        plt.grid()
        plt.savefig(filenames[1])
        plt.clf()

    #################
    # TESTING PLOTS #
    #################

    def check_truth_label_vs_pdgID(self, final_val_results):
        class_pdgID_pairs = set(zip(final_val_results['pdgID'], final_val_results['class_truth']))
        assert len(class_pdgID_pairs) == 2

    def plot_score_bins(self, final_val_results, filename):
        raw_score = final_val_results['class_raw_prediction'][()]
        class_truth = final_val_results['class_truth'][()]
        plt.hist(raw_score[class_truth==0], bins=50, range=(0,1), histtype='step', color='r', label=final_val_results['pdgID'][class_truth==0][0])
        plt.hist(raw_score[class_truth==1], bins=50, range=(0,1), histtype='step', color='b', label=final_val_results['pdgID'][class_truth==1][0])
        plt.title('Raw Net Score')
        plt.xlabel('score')
        plt.grid()
        plt.legend()
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

        # test plots
        self.check_truth_label_vs_pdgID(final_val_results)
        self.plot_score_bins(final_val_results, folder+"/score.eps")

        # final classification plots
        self.plot_ROC(final_val_results, folder+"/ROC.eps")

        self.plot_accuracy_bins('energy', final_val_results, folder+"/accuracy_vs_energy.eps")
        self.plot_accuracy_bins('eta', final_val_results, folder+"/accuracy_vs_eta.eps")
        self.plot_accuracy_bins('openingAngle', final_val_results, folder+"/accuracy_vs_openingAngle.eps")

        self.plot_history(test_train_history['class_reg_loss_train_batch'], test_train_history['class_reg_loss_test_batch'], loss=True, batch=True, filename=folder+"/loss_batches.eps")
        self.plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename=folder+"/accuracy_batches.eps")
        self.plot_history(test_train_history['class_reg_loss_train_epoch'], test_train_history['class_reg_loss_test_epoch'], loss=True, batch=False, filename=folder+"/loss_epoches.eps")
        self.plot_history(test_train_history['class_acc_train_epoch'], test_train_history['class_acc_test_epoch'], loss=False, batch=False, filename=folder+"/accuracy_epoches.eps")

        # regression plots
        if 'reg_energy_prediction' in final_val_results.keys():
            self.plot_regression_bins('energy', 'energy', final_val_results,
                                      [folder+'/reg_bias_energy_vs_energy.eps', folder+'/reg_res_energy_vs_energy.eps'])
            self.plot_regression_bins('eta', 'energy', final_val_results,
                                      [folder+'/reg_bias_energy_vs_eta.eps', folder+'/reg_res_energy_vs_eta.eps'])
            self.plot_regression_bins('phi', 'energy', final_val_results,
                                      [folder+'/reg_bias_energy_vs_phi.eps', folder+'/reg_res_energy_vs_phi.eps'])
        if 'reg_eta_prediction' in final_val_results.keys():
            self.plot_regression_bins('energy', 'eta', final_val_results,
                                      [folder+'/reg_bias_eta_vs_energy.eps', folder+'/reg_res_eta_vs_energy.eps'])
            self.plot_regression_bins('eta', 'eta', final_val_results,
                                      [folder+'/reg_bias_eta_vs_eta.eps', folder+'/reg_res_eta_vs_eta.eps'])
            self.plot_regression_bins('phi', 'eta', final_val_results,
                                      [folder+'/reg_bias_eta_vs_phi.eps', folder+'/reg_res_eta_vs_phi.eps'])

    def analyze_online(self, history, folder): 
        """
        Online plotting tool for classifier results representation
        """
        TRAIN, _, TEST = 0, 1, 2
        # ['class_reg_loss', ..., 'class_acc'] = [0, ..., 5]
        BATCH, EPOCH = 0, 1
        self.plot_history(np.array(history[0][0][0]), np.array(history[0][2][0]), loss=True, batch=True, filename=folder+"/loss_batches.eps")
        self.plot_history(np.array(history[5][0][0]), np.array(history[5][2][0]), loss=False, batch=True, filename=folder+"/accuracy_batches.eps")
        self.plot_history(np.array(history[0][0][1]), np.array(history[0][2][1]), loss=True, batch=False, filename=folder+"/loss_epoches.eps")
        self.plot_history(np.array(history[5][0][1]), np.array(history[5][2][1]), loss=False, batch=False, filename=folder+"/accuracy_epoches.eps")

