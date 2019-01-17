from __future__ import division
import numpy as np
import sys
import h5py as h5
import os
import collections
import glob

# list all features in tree
def h5_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if prefix=='': path = key
        if isinstance(item, h5.Dataset):
            yield path
        elif isinstance(item, h5.Group):
            for data in h5_dataset_iterator(item, path): yield data

def normalizeFiles(inFile, outFile):

    files=sorted(glob.glob(inFile))
    totalFeatures = collections.OrderedDict()

    for file_index in range(len(files)):
        f1 = h5.File(files[file_index], 'r')
	features = []
	for path in h5_dataset_iterator(f1): features.append(path)
        for feature in features:
            if(feature!='ECAL' and feature !='HCAL' and feature!='pdgID' and feature!='conversion' and feature!='energy' and feature!='bestJets10' and feature!='bestJets11' and feature!='bestJets20' and feature!= 'bestJets20' and feature!='bestJets21' and feature!='tau1' and feature!='tau2' and feature!='tau3' and feature!='tau2_over_tau1' and feature!='tau3_over_tau2'):
                features1=np.array(f1[feature][:])
                features1[features1==np.inf]=0
                features1[features1==-np.inf]=0
                features1=np.nan_to_num(features1)
                if file_index == 0:
                    totalFeatures[feature] = features1
                else:
                    totalFeatures[feature] = np.concatenate((totalFeatures[feature], features1))
        f1.close()

    means = collections.OrderedDict()
    stds = collections.OrderedDict()

    for feature in totalFeatures.keys():
        means[feature] = np.mean(totalFeatures[feature])
        stds[feature] = np.std(totalFeatures[feature])

    for file_index in range(len(files)):
        # Save features to an h5 file
        f1 = h5.File(files[file_index], 'r')
        f2 = h5.File(outFile+os.path.basename(files[file_index]), "w")
        for feature in totalFeatures:
            features1=np.array(f1[feature][:])
            features1[features1==np.inf]=0
            features1[features1==-np.inf]=0
            features1=np.nan_to_num(features1)
            f2.create_dataset(feature,data=(features1 - means[feature])/stds[feature])
        f1.close()
        f2.close()
    
#################
# Main function #
#################

if __name__ == "__main__":
    import sys
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    normalizeFiles(inFile, outFile)
