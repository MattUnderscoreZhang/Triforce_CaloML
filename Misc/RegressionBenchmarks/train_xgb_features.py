import os,sys
import h5py as h5
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if len(sys.argv) < 2:
    print 'usage: python train_xgb_features.py <output_label>'

label = sys.argv[1]

## local laptop
basepath = '/home/olivito/datasci/lcd/data'
## culture-plate at caltech
#basepath = '/data/shared/LCDLargeWindow'

input_filename = basepath + '/fixedangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/fixedangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/fixedangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/fixedangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

# if test_filename == input_filename, will use the same file for training and testing with a 70/30 split
# if test_filename is another file, will use one full file for training and the other for testing
test_filename = input_filename
#test_filename = basepath + '/fixedangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

do_transverse = False

def load_hdf5(filename,do_transverse = False):
    with h5.File(filename, 'r') as f:
        ECAL_E = f['ECAL_E'][:].reshape(-1,1)
        HCAL_E = f['HCAL_E'][:].reshape(-1,1)
        conversion = f['conversion'][:].reshape(-1,1)
        ECALmomentX2 = f['ECALmomentX2'][:].reshape(-1,1)
        ECALmomentY2 = f['ECALmomentY2'][:].reshape(-1,1)
        ECALmomentZ1 = f['ECALmomentZ1'][:].reshape(-1,1)
        ECALmomentXY2 = np.sqrt(np.square(ECALmomentX2) + np.square(ECALmomentY2))
        HCALmomentX2 = f['HCALmomentX2'][:].reshape(-1,1)
        HCALmomentY2 = f['HCALmomentY2'][:].reshape(-1,1)
        HCALmomentXY2 = np.sqrt(np.square(HCALmomentX2) + np.square(HCALmomentY2))
        HCALmomentZ1 = f['HCALmomentZ1'][:].reshape(-1,1)
        energy = f['energy'][:].reshape(-1,1)
        eta = np.zeros_like(energy, dtype=np.float32)
        if 'eta' in f.keys():
            eta = f['eta'][:].reshape(-1,1)
            if do_transverse:
                ECAL_E = ECAL_E/np.cosh(eta)
                HCAL_E = HCAL_E/np.cosh(eta)
                energy = energy/np.cosh(eta)

        ## select which features to use
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentZ1, HCALmomentXY2, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, HCALmomentXY2], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, HCALmomentXY2, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentY2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentXY2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2], axis=1)
        features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1], axis=1) ## default fixed angle baseline
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1, eta], axis=1) ## default variable angle baseline
        #features = np.concatenate([ECAL_E, HCAL_E, eta], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E], axis=1)
    return features.astype(np.float32), energy.astype(np.float32)

# for ChPi: remove events with reco energy < 0.3 * true energy
def clean_chpi(X,y):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((Xy[:,0] + Xy[:,1]) > 0.3 * Xy[:,-1])]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    print X.shape, y.shape
    return X, y

# for variable angle electrons, photons, pi0s:
#  remove events with reco energy < 0.66 * true energy or H/E > 0.4
def clean_varangle(X,y):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((Xy[:,0] + Xy[:,1]) > 0.66 * Xy[:,-1])]
    Xy_good = Xy_good[np.where((Xy_good[:,1]/Xy_good[:,0]) < 0.4)]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    print X.shape, y.shape
    return X, y

# apply a min/max energy cut.  Assumes that energy is column -1...
def energy_cut(X,y,cutmin = -1,cutmax = 1000):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((Xy[:,-1] > cutmin) & (Xy[:,-1] < cutmax))]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    print X.shape, y.shape
    return X, y

# apply a min/max eta cut.  Assumes that eta is column -2...
def abseta_cut(X,y,cutmin = -1,cutmax = 1000):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((np.fabs(Xy[:,-2]) > cutmin) & (np.fabs(Xy[:,-2]) < cutmax))]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    print X.shape, y.shape
    return X, y

# select only photons that have (not) converted.  Assumes that conversion is column -2...
def select_conversions(X,y,cut = 1):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where(Xy[:,-2] == cut)]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    print X.shape, y.shape
    return X, y

#X, y = load_hdf5(input_filename)
X, y = load_hdf5(input_filename,do_transverse=do_transverse)
print X.shape, y.shape
if 'ChPi' in input_filename: X,y = clean_chpi(X,y)
elif 'varangle' in input_filename: X,y = clean_varangle(X,y)

#X,y = abseta_cut(X,y,cutmax=0.05)
## remove eta var if necessary
#X = X[:,:-1]

# ## select only converted photons
# X, y = select_conversions(X, y, 1)
# ## remove conversion variable
# X = X[:,:-1]

X_train, X_test, y_train, y_test = None, None, None, None

# if only using one file: split into train/test
if test_filename == input_filename:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# if using two files: shuffle within each file
else:
    X_train, y_train = shuffle(X, y)
    X_test, y_test = load_hdf5(test_filename,do_transverse=do_transverse)
    if 'ChPi' in test_filename: X_test, y_test = clean_chpi(X_test, y_test)
    X_test, y_test = shuffle(X_test, y_test)

# ## select only converted photons for test set
# X_test, y_test = select_conversions(X_test, y_test, 1)
# ## remove conversion variable
# X_train = X_train[:,:-1]
# X_test = X_test[:,:-1]

## if cutting on energy for training set only
#X_train,y_train = energy_cut(X_train,y_train,cutmax=400)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 3, 'objective': 'reg:linear','eval_metric':['rmse']}
## for hyperparameter scans
#param = {'max_depth': 10, 'objective': 'reg:linear','eval_metric':['rmse'],'min_child_weight': 5}
#param = {'max_depth': 3, 'objective': 'reg:linear','eval_metric':['rmse'],'min_child_weight': 1,'eta':0.9}
evallist = [(dtrain, 'train'), (dtest, 'test')]

num_round = 1000
progress = {}

bst = xgb.train(param,dtrain,num_round,evallist,evals_result=progress,early_stopping_rounds=10)

y_pred = bst.predict(dtest)

output_dir = './Output/%s/'%(label)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = output_dir + 'results.h5'
outfile = h5.File(output_filename,'w')

outfile.create_dataset('reg_loss_history_train',data=np.asarray(progress['train']['rmse']))
outfile.create_dataset('reg_loss_history_test',data=np.asarray(progress['test']['rmse']))
outfile.create_dataset('reg_energy_prediction',data=np.asarray(y_pred))
outfile.create_dataset('energy',data=np.asarray(y_test))

outfile.close()

bst.save_model(output_dir+'model.xgb')
