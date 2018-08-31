import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

##############
# ROC Curves #
##############

# final_val_results = h5.File("/data/LCD/NewSamples/HyperparameterResults/DNN/DNN_Output_4_512_0.0002_0.04_Scan1/validation_results.h5")
final_val_results = h5.File("/data/LCD/NewSamples/HyperparameterResults/DNN/DNN_Output_4_512_0.0004_0.08_Scan1/validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='DNN (area = %0.2f)' % roc_auc)

# final_val_results = h5.File("/data/LCD/NewSamples/HyperparameterResults/CNN/CNN_Output_4_512_0.0004_0.12_6_6_Scan1/validation_results.h5")
final_val_results = h5.File("/data/LCD/NewSamples/HyperparameterResults/CNN/CNN_Output_3_512_0.0004_0.08_6_6_Scan1/validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', lw=2, label='CNN (area = %0.2f)' % roc_auc)

final_val_results = h5.File("/data/LCD/NewSamples/HyperparameterResults/GN/GN_Output_1024_0.0001_0.01_Scan1/validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='red', lw=2, label='GN (area = %0.2f)' % roc_auc)

# data = h5.File("BDT_Results.h5")
# fpr = data['fpr']
# tpr = data['tpr']
# roc_auc = metrics.auc(data['fpr'], data['tpr'])
# plt.plot(fpr, tpr, color='blue', lw=2, label='BDT (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid('on', linestyle='--')
plt.title('ROC Curve for Classification')
plt.legend(loc="lower right")
plt.savefig("compare_ROC.png")
