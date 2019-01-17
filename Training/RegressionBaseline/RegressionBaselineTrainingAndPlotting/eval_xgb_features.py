import os,sys
import h5py as h5
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if len(sys.argv) < 2:
    print 'usage: python eval_xgb_features.py <output_label>'

label = sys.argv[1]

## local laptop
basepath = '/home/olivito/datasci/lcd/data'
## culture-plate at caltech
#basepath = '/data/shared/LCDLargeWindow'

#test_filename = basepath + '/fixedangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
test_filename = basepath + '/fixedangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

#training_filename = 'Output/Reg_TrainGammaFixed_TestEleFixed_xgb_ECALZ1Only_depth3_1000rounds/model.xgb'
training_filename = 'Output/Reg_EleFixed_xgb_ECALZ1Only_depth3_1000rounds/model.xgb'

def load_hdf5(filename):
    with h5.File(filename, 'r') as f:
        ECAL_E = f['ECAL_E'][:].reshape(-1,1)
        HCAL_E = f['HCAL_E'][:].reshape(-1,1)
        ECALmomentX2 = f['ECALmomentX2'][:].reshape(-1,1)
        ECALmomentY2 = f['ECALmomentY2'][:].reshape(-1,1)
        ECALmomentZ1 = f['ECALmomentZ1'][:].reshape(-1,1)
        ECALmomentXY2 = np.sqrt(np.square(ECALmomentX2) + np.square(ECALmomentY2))
        HCALmomentX2 = f['HCALmomentX2'][:].reshape(-1,1)
        HCALmomentY2 = f['HCALmomentY2'][:].reshape(-1,1)
        HCALmomentXY2 = np.sqrt(np.square(HCALmomentX2) + np.square(HCALmomentY2))
        HCALmomentZ1 = f['HCALmomentZ1'][:].reshape(-1,1)
        try:
            eta = f['eta'][:].reshape(-1,1)
        except:
            pass
        ## select features to use.  Must match those used to train the model
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentZ1, HCALmomentXY2, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, HCALmomentXY2], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, HCALmomentXY2, HCALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentY2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentXY2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2, ECALmomentZ1], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentX2], axis=1)
        features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1], axis=1) ## default fixed angle baseline
        #features = np.concatenate([ECAL_E, HCAL_E, ECALmomentZ1, eta], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E, eta], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E], axis=1)
        energy = f['energy'][:]
    return features.astype(np.float32), energy.astype(np.float32)

# for ChPi: remove events with reco energy < 0.3 * true energy
def clean_chpi(X,y):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((Xy[:,0] + Xy[:,1]) > 0.3 * Xy[:,-1])]
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

X_test, y_test = load_hdf5(test_filename)
print X_test.shape, y_test.shape
if 'ChPi' in test_filename: X_test,y_test = clean_chpi(X_test,y_test)

X_test,y_test = abseta_cut(X_test,y_test,cutmax=0.05)
## remove eta column if necessary
#X_test = X_test[:,:-1]

dtest = xgb.DMatrix(X_test, label=y_test)

## load previously trained model
bst = xgb.Booster()
bst.load_model(training_filename)

y_pred = bst.predict(dtest)

output_dir = './Output/%s/'%(label)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = output_dir + 'results.h5'
outfile = h5.File(output_filename,'w')

outfile.create_dataset('regressor_pred',data=np.array(y_pred))
outfile.create_dataset('regressor_true',data=np.array(y_test))

outfile.close()
