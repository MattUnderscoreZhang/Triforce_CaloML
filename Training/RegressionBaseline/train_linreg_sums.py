import os,sys,glob,pdb
import h5py as h5
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

## local laptop
# basepath = '/home/olivito/datasci/lcd/data'
## culture-plate at caltech
#basepath = '/data/shared/LCDLargeWindow'

# input_filename = basepath + '/fixedangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/fixedangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/fixedangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/fixedangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#input_filename = basepath + '/varangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

# if test_filename == None, will use the same file for training and testing with a 70/30 split
# if test_filename is another file, will use one full file for training and the other for testing
#test_filename = basepath + '/fixedangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/fixedangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/EleEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/GammaEscan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/Pi0Escan/merged_featuresonly/merged_minfeatures.h5'
#test_filename = basepath + '/varangle/ChPiEscan/merged_featuresonly/merged_minfeatures.h5'

## UTA
test_filename = None

# input_filepaths = '/public/data/calo/RandomAngle/CMS/Gamma/*.h5'
# output_folder = 'GammaLinReg_CMS'

# input_filepaths = '/public/data/calo/RandomAngle/CMS/Pi0/*.h5'
# output_folder = 'Pi0LinReg_CMS'

# input_filepaths = '/public/data/calo/RandomAngle/ATLAS/Gamma/*.h5'
# output_folder = 'GammaLinReg_ATLAS'

input_filepaths = '/public/data/calo/RandomAngle/ATLAS/Pi0/*.h5'
output_folder = 'Pi0LinReg_ATLAS'

def load_hdf5(filename):
    with h5.File(filename, 'r') as f:
        ECAL_E = f['ECAL_E'][:].reshape(-1,1)
        HCAL_E = f['HCAL_E'][:].reshape(-1,1)
        features = np.concatenate([ECAL_E, HCAL_E], axis=1)
        energy = f['energy'][:]
    return features.astype(np.float32), energy.astype(np.float32)

# for ChPi: remove events with reco energy < 0.3 * true energy
def clean_chpi(X,y):
    Xy = np.concatenate([X,y.reshape(-1,1)],axis=1)
    Xy_good = Xy[np.where((Xy[:,0] + Xy[:,1]) > 0.3 * Xy[:,-1])]
    X, y = Xy_good[:,:-1], Xy_good[:,-1]
    return X, y

input_filenames = glob.glob(input_filepaths)
X = None
y = None
for input_filename in input_filenames:
    if X is None:
        X, y = load_hdf5(input_filename)
        if 'ChPi' in input_filename: X,y = clean_chpi(X,y)
    else:
        this_X, this_y = load_hdf5(input_filename)
        if 'ChPi' in input_filename: this_X,this_y = clean_chpi(this_X,this_y)
        X = np.concatenate([X, this_X])
        y = np.concatenate([y, this_y])

X_train, X_test, y_train, y_test = None, None, None, None

# if only using one file: split into train/test
if test_filename == None:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# if using two files: shuffle within each file
else:
    X_train, y_train = shuffle(X, y)
    X_test, y_test = load_hdf5(test_filename)
    if 'ChPi' in test_filename: X_test, y_test = clean_chpi(X_test, y_test)
    X_test, y_test = shuffle(X_test, y_test)

linreg = LinearRegression()
linreg.fit(X_train,y_train)

print(linreg.coef_)

y_pred = linreg.predict(X_test)

output_dir = './Output/%s/'%(output_folder)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = output_dir + 'results.h5'
outfile = h5.File(output_filename,'w')

outfile.create_dataset('regressor_pred',data=np.array(y_pred))
outfile.create_dataset('regressor_true',data=np.array(y_test))

outfile.close()

with open(output_dir+'coefs.txt','w') as coefs_file:
    coefs_file.write(str(linreg.coef_)+'\n')
