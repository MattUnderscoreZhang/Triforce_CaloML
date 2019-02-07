import matplotlib
matplotlib.use('Agg') # NOQA
import h5py as h5
import matplotlib.pyplot as plt
from sklearn import metrics
import pathlib

##############
# File Paths #
##############

base_path = "/home/matt/Projects/calo/FinalResults/Classification/"
DNN_path = base_path + "DNN/"
CNN_path = base_path + "CNN/"
GN_path = base_path + "GN/"
BDT_path = base_path + "BDT/"
out_path = "/home/matt/Projects/calo/Analysis/Plots/Classification/"
log_scale = False


############
# Plotting #
############

def plot_history(train, test, loss, batch, filename):

    downsampling = 50
    train = train[::downsampling]
    test = test[::downsampling]
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
    plt.clf()


def plot_all_history_curves(geometry=""):

    if DNN_path is not None:
        test_train_history = h5.File(DNN_path + geometry + "training_results.h5")
        plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename=out_path + geometry + "DNN_accuracy_batches.eps")

    if CNN_path is not None:
        test_train_history = h5.File(CNN_path + geometry + "training_results.h5")
        plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename=out_path + geometry + "CNN_accuracy_batches.eps")

    if GN_path is not None:
        test_train_history = h5.File(GN_path + geometry + "training_results.h5")
        plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename=out_path + geometry + "GN_accuracy_batches.eps")


def plot_all_ROC_curves(geometry=""):

    if DNN_path is not None:
        final_val_results = h5.File(DNN_path + geometry + "validation_results.h5")
        truth = final_val_results['class_truth']
        scores = final_val_results['class_raw_prediction']
        fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='DNN (area = %0.2f)' % roc_auc)

    if CNN_path is not None:
        final_val_results = h5.File(CNN_path + geometry + "validation_results.h5")
        truth = final_val_results['class_truth']
        scores = final_val_results['class_raw_prediction']
        fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='green', lw=2, label='CNN (area = %0.2f)' % roc_auc)

    if GN_path is not None:
        final_val_results = h5.File(GN_path + geometry + "validation_results.h5")
        truth = final_val_results['class_truth']
        scores = final_val_results['class_raw_prediction']
        fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='red', lw=2, label='GN (area = %0.2f)' % roc_auc)

    data = h5.File(BDT_path + geometry + "Results.h5")
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = metrics.auc(data['fpr'], data['tpr'])
    plt.plot(fpr, tpr, color='blue', lw=2, label='BDT (area = %0.2f)' % roc_auc)

    if log_scale:
        plt.yscale('log')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    if log_scale:
        plt.xlim([0.0, 0.3])
        plt.ylim([0.9, 1.01])
    else:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on', linestyle='--')
    plt.title('ROC Curve for Classification')
    plt.legend(loc="lower right")
    plt.savefig(out_path + geometry + "compare_ROC.eps")


########
# Main #
########

if __name__ == "__main__":

    CNN_path = None
    GN_path = None
    for geometry in ["CLIC/", "ATLAS/", "CMS/"]:
        if pathlib.Path(out_path + geometry).exists() is False:
            pathlib.Path(out_path + geometry).mkdir(parents=True)
        plot_all_history_curves(geometry)
        plot_all_ROC_curves(geometry)
