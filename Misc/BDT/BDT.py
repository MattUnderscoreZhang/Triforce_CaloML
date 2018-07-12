import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
import numpy as np
import glob
import os
import sys
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
import cPickle

###############
# Set options #
###############

# basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/V4/Original/EleChPi/"
# samplePath = [basePath + "ChPiEscan/ChPiEscan_*.h5", basePath + "EleEscan/EleEscan_*.h5"]
# target_names = ['charged pion', 'electron']
# classPdgID = [211, 11] # absolute IDs corresponding to paths above
basePath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/V4/Original/GammaPi0/"
samplePath = [basePath + "Pi0Escan/Pi0Escan_*.h5", basePath + "GammaEscan/GammaEscan_*.h5"]
target_names = ['neutral pion', 'photon']
classPdgID = [111, 22] # absolute IDs corresponding to paths above

badKeys = ['ECALmomentX1', 'ECALmomentY1', 'HCALmomentX1', 'HCALmomentY1', 'ECAL/ECAL', 'HCAL/HCAL', 'Event/conversion', 'Event/energy', 'Event/px', 'Event/py', 'Event/pz', 'N_Subjettiness/bestJets1', 'N_Subjettiness/bestJets2'] # leave pdgID for now - needed below

OutPath = "/u/sciteam/zhang10/Projects/DNNCalorimeter/SubmissionScripts/BDT/"+sys.argv[1]
max_depth = int(sys.argv[2]) # 3
n_estimators = int(sys.argv[3]) # 800
learning_rate = float(sys.argv[4]) # 0.5

##########################
# Load and prepare files #
##########################

# load files
print "Loading files"
dataFileNames = []
for particlePath in samplePath:
    dataFileNames += glob.glob(particlePath)

dataFiles = []
for i in range(len(dataFileNames)):
    if os.path.exists(dataFileNames[i]):
        dataFiles.append(h5.File(dataFileNames[i], "r"))

# list all features in tree
print "Finding features"
features = []
def h5_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if prefix=='': path = key 
        if isinstance(item, h5.Dataset):
            yield path
        elif isinstance(item, h5.Group):
            for data in h5_dataset_iterator(item, path): yield data
for path in h5_dataset_iterator(dataFiles[0]):
    features.append(path)

# remove features bad for BDT
for key in badKeys:
    if key in features: features.remove(key)

# concat all data to form X and y
data = []
print "Reading features"
for count, feature in enumerate(features):
    sys.stdout.flush()
    newFeature = dataFiles[0][feature]
    for fileN in range(1, len(dataFiles)):
        newFeature = np.concatenate((newFeature, dataFiles[fileN][feature]))
    if feature == 'pdgID':
        y = newFeature
        for i, ID in enumerate(classPdgID):
            y[y==ID] = i
    else:
        data.append(newFeature);
features.remove('pdgID')
features = np.array(features)
features = features.astype(str)

X = np.column_stack(data)
y = y[np.isfinite(X).all(axis=1)]
X = X[np.isfinite(X).all(axis=1)]

# split test and train
# X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=0.33, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33, random_state=492)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=492)

#############
# Train BDT #
#############

bdt = GradientBoostingClassifier(max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        verbose=True,
        random_state=492)

print "Training BDT"
bdt.fit(X_train, y_train)
print "Analyzing BDT results"
y_predicted = bdt.predict(X_test)
decisions = bdt.decision_function(X_test)
print (classification_report(y_test, y_predicted, target_names=target_names, digits=4))
print ("Area under ROC curve: %.4f"%(roc_auc_score(y_test, decisions)))

################
# Plot results #
################

# save file
if not os.path.exists(OutPath): os.makedirs(OutPath)
file = h5.File(OutPath+"Results.h5", 'w')

# Precision (P) is defined as the number of true positives (T_p) over the number of true positives plus the number of false positives (F_p).  
# P = \frac{T_p}{T_p+F_p}  
# R = \frac{T_p}{T_p + F_n}

# compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.4f)'%(roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(OutPath+'ROC.png', bbox_inches='tight')
plt.cla(); plt.clf()

file.create_dataset("roc_auc", data=np.array([roc_auc]))
file.create_dataset("fpr", data=np.array(fpr))
file.create_dataset("tpr", data=np.array(tpr))
file.create_dataset("thresholds", data=np.array(thresholds))

# train-test plots
def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0], color='r', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', normed=True, label='S (train)')
    plt.hist(decisions[1], color='b', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', normed=True, label='B (train)')

    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')

    plt.savefig(OutPath+'compare_train_test.png', bbox_inches='tight')
    
compare_train_test(bdt, X_train, y_train, X_test, y_test)

with open(OutPath+'bdt.pkl', 'wb') as pickleFile:
    cPickle.dump(bdt, pickleFile)    

# feature rankings
importances = bdt.feature_importances_
std = np.std([tree[0].feature_importances_ for tree in bdt.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

file.create_dataset("features", data=features)
file.create_dataset("importances", data=np.array(importances))
file.create_dataset("std", data=np.array(std))

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f+1, features[indices[f]], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]))
plt.xlim([-1, X.shape[1]])
plt.savefig(OutPath+'feature_importances.png', bbox_inches='tight')
