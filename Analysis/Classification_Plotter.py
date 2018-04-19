import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from triforce_helper_functions import eval
import numpy as np
from sklearn import metrics
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
        plt.plot(range(train.shape[0])+1, train, 'o-', color="g", label="Training Loss", alpha=0.5)
        plt.plot(range(test.shape[0])+1, test, 'o-', color="r", label="Test Loss", alpha=0.5)
        plt.legend(loc="best")
        plt.savefig(filename)

    def plot_accuracy_vs_epoch(self, train, test, filename): 

        plt.figure()
        plt.title("Classifier Accuracy vs Epoch")
        plt.xlabel("Training Epoch")
        plt.ylabel("Accuracy")
        plt.grid('on', linestyle='--')
        plt.plot(range(train.shape[0])+1, train, 'o-', color="g", label="Training Accuracy", alpha=0.5)
        plt.plot(range(test.shape[0])+1, test, 'o-', color="r", label="Test Accuracy", alpha=0.5)
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

    def plot_ROC(self, outputs, truth, filename):

        fpr, tpr, thresholds = metrics.roc_curve(truth, outputs)
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

    def analyze(self, tools, testLoader, out_file):

        [classifier, regressor, GAN] = tools

        classifier_test_results = []
        for data in testLoader:
            ECALs, HCALs, ys, energies = data
            ECALs, HCALs, ys, energies = Variable(ECALs.cuda()), Variable(HCALs.cuda()), Variable(ys.cuda()), Variable(energies.cuda())
            if (classifier != None): classifier_test_results.append(eval(classifier, ECALs, HCALs, ys))
            else: classifier_test_results.append((0,0,0,0,0,0))

        n_test_batches = len(classifier_test_results)
        # extract test loss
        classifier_test_loss = [classifier_test_results[i][0] for i in range(len(classifier_test_results))]
        classifier_test_loss = sum(classifier_test_loss) / n_test_batches
        # extract test accuracy
        classifier_test_accuracy = [classifier_test_results[i][1] for i in range(len(classifier_test_results))]
        classifier_test_accuracy = sum(classifier_test_accuracy) / n_test_batches
        # extract test outputs
        classifier_test_outputs = torch.FloatTensor([])
        for i in range(len(classifier_test_results)):
            if i == 0: 
                classifier_test_outputs = list(classifier_test_results[0])[2][:,1]
            else: 
                classifier_test_outputs = torch.cat((classifier_test_outputs, list(classifier_test_results[i])[2][:,1]), 0)
        classifier_test_outputs = np.array(classifier_test_outputs)
        # extract test truth
        classifier_test_truth = torch.FloatTensor([])
        for i in range(len(classifier_test_results)): 
            if i == 0: 
                classifier_test_truth = classifier_test_results[i][3]
            else: 
                classifier_test_truth = torch.cat((classifier_test_truth, classifier_test_results[i][3]))
        classifier_test_truth = np.array(classifier_test_truth)
        

        print('test loss: (C) %.4f; test accuracy: (C) %.4f' % (classifier_test_loss, classifier_test_accuracy))
        # if (classifier != None): out_file.create_dataset("classifier_test_results", data=classifier_test_results)
        if (classifier != None): out_file.create_dataset("classifier_test_accuracy", data=classifier_test_accuracy) 

        folder = out_file.filename[:out_file.filename.rfind('/')]
        self.plot_loss_vs_batches(out_file['loss_classifier_train_batch'].value, out_file['loss_classifier_test_batch'].value, folder+"/loss_batches.png")
        self.plot_accuracy_vs_batches(out_file['accuracy_classifier_train_batch'].value, out_file['accuracy_classifier_test_batch'].value, folder+"/accuracy_batches.png")
        self.plot_loss_vs_epoch(out_file['loss_classifier_train_epoch'].value, out_file['loss_classifier_test_epoch'].value, folder+"/loss_epoch_batches.png")
        self.plot_accuracy_vs_epoch(out_file['accuracy_classifier_train_epoch'].value, out_file['accuracy_classifier_test_epoch'].value, folder+"/accuracy_epoch_batches.png")
        # try: 
        self.plot_ROC(classifier_test_outputs, classifier_test_truth, folder+"/ROC.png")
