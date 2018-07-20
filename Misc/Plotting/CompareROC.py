import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

final_val_results = h5.File("DNN_validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='DNN (area = %0.2f)' % roc_auc)

final_val_results = h5.File("CNN_validation_results.h5")
truth = final_val_results['class_truth']
scores = final_val_results['class_raw_prediction']
fpr, tpr, thresholds = metrics.roc_curve(truth, scores)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', lw=2, label='CNN (area = %0.2f)' % roc_auc)

data = h5.File("BDT_Results.h5")
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
plt.savefig("compare_ROC.png")
