import matplotlib
matplotlib.use('Agg')
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

##############
# File Paths #
##############

base_path = "/home/mazhang/Triforce_CaloML/FinalResults/CLIC/GammaPi0/TriForce/"
DNN_path = base_path + "Output_DNN_4_512_0.0002_0.04/"
CNN_path = base_path + "Output_CNN_4_512_0.0004_0.12/"
GN_path = base_path + "Output_GN_0_1024_0.0001_0.01/"
BDT_path = "/home/mazhang/Triforce_CaloML/FinalResults/CLIC/GammaPi0/BDT/GammaPi0_WithR9/Results.h5"
out_path = "Plots/GammaPi0/"

###################
# Training Curves #
###################

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

test_train_history = h5.File(DNN_path + "training_results.h5")
plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename = out_path + "DNN_accuracy_batches.png")

test_train_history = h5.File(CNN_path + "training_results.h5")
plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename = out_path + "CNN_accuracy_batches.png")

test_train_history = h5.File(GN_path + "training_results.h5")
plot_history(test_train_history['class_acc_train_batch'], test_train_history['class_acc_test_batch'], loss=False, batch=True, filename = out_path + "GN_accuracy_batches.png")

##############
# ROC Curves #
##############

final_val_results = h5.File(DNN_path + "validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='DNN (area = %0.2f)' % roc_auc)

final_val_results = h5.File(CNN_path + "validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', lw=2, label='CNN (area = %0.2f)' % roc_auc)

final_val_results = h5.File(GN_path + "validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='red', lw=2, label='GN (area = %0.2f)' % roc_auc)

data = h5.File(BDT_path)
fpr = data['fpr']
tpr = data['tpr']
roc_auc = metrics.auc(data['fpr'], data['tpr'])
plt.plot(fpr, tpr, color='blue', lw=2, label='BDT (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid('on', linestyle='--')
plt.title('ROC Curve for Classification')
plt.legend(loc="lower right")
plt.savefig(out_path + "compare_ROC.png")
