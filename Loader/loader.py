# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np

def load_hdf5(file, pdgIDs, loadMinimalFeatures=None):
    '''Loads H5 file. Used by HDF5Dataset.'''
    return_data = {}
    with h5py.File(file, 'r') as f:
        # (default) load full ECAL / HCAL arrays and standard features
        if loadMinimalFeatures is None:
            return_data['ECAL'] = f['ECAL'][:].astype(np.float32)
            n_events = len(return_data['ECAL'])
            return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
            return_data['ECAL_E'] = f['ECAL_E'][:].astype(np.float32)
            return_data['HCAL_E'] = f['HCAL_E'][:].astype(np.float32)
            return_data['HCAL_ECAL_ERatio'] = f['HCAL_ECAL_ERatio'][:].astype(np.float32)
            return_data['pdgID'] = f['pdgID'][:].astype(int)
            return_data['classID'] = np.array([pdgIDs[abs(i)] for i in return_data['pdgID']]) # PyTorch expects class index instead of one-hot
            if 'energy' in f.keys():
                return_data['energy'] = f['energy'][:].astype(np.float32)
            else: return_data['energy'] = np.zeros(n_events, dtype=np.float32)
            if 'eta' in f.keys():
                return_data['eta'] = f['eta'][:].astype(np.float32)
            else: return_data['eta'] = np.zeros(n_events, dtype=np.float32)
        # minimal data load: only load specific features that are requested
        else:
            for feat in loadMinimalFeatures:
                return_data[feat] = f[feat][:]
    return return_data

def load_3d_hdf5(file, pdgIDs):
    '''Loads H5 file and adds an extra dimension for CNN. Used by HDF5Dataset.'''
    return_data = load_hdf5(file, pdgIDs)
    return_data['ECAL'] = np.expand_dims(return_data['ECAL'], axis=1)
    return_data['HCAL'] = np.expand_dims(return_data['HCAL'], axis=1)
    return return_data

class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Used to create PyTorch DataLoader.
    Arguments:
        dataname_tuples: list of filename tuples, where each tuple will be mixed into a single file
        num_per_file: number of events in each data file
    """

    def __init__(self, dataname_tuples, pdgIDs, filters = []):
        self.dataname_tuples = sorted(dataname_tuples)
        self.nClasses = len(dataname_tuples[0])
        self.num_per_file = len(dataname_tuples) * [0]
        self.fileInMemory = -1
        self.fileInMemoryFirstIndex = 0
        self.fileInMemoryLastIndex = -1
        self.data = {}
        self.pdgIDs = {}
        self.filters = filters
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.countEvents()

    def countEvents(self):
        # make minimal list of inputs needed to check (or count events)
        minFeatures = []
        for filt in self.filters: minFeatures += filt.featuresUsed
        # use energy to count number of events if no filters
        if len(minFeatures) == 0: minFeatures.append('energy')
        totalevents = 0
        for fileN in range(len(self.dataname_tuples)):
            nevents_before, nevents_after = [], []
            for dataname in self.dataname_tuples[fileN]:
                file_data = load_hdf5(dataname, self.pdgIDs, minFeatures)
                nevents_before.append(len(list(file_data.values())[0]))
                for filt in self.filters: filt.filter(file_data)
                nevents_after.append(len(list(file_data.values())[0]))
            self.num_per_file[fileN] = min(nevents_after) * self.nClasses
            totalevents += min(nevents_before) * self.nClasses
        print('total events:',totalevents)
        if len(self.filters) > 0:
            print('total events passing filters:',sum(self.num_per_file))

    def __getitem__(self, index):
        # if we started to look at a new file, read the file data
        if(index > self.fileInMemoryLastIndex):
            # update indices to new file
            self.fileInMemory += 1
            self.fileInMemoryFirstIndex = int(self.fileInMemoryLastIndex+1)
            self.fileInMemoryLastIndex += self.num_per_file[self.fileInMemory]
            self.data = {}
            for dataname in self.dataname_tuples[self.fileInMemory]:
                file_data = load_hdf5(dataname, self.pdgIDs)
                # apply any filters here
                if not self.filters is None:
                    for filt in self.filters: filt.filter(file_data)
                for key in file_data.keys():
                    if key in self.data.keys():
                        self.data[key] = np.append(self.data[key], file_data[key], axis=0)
                    else:
                        self.data[key] = file_data[key]
        # return the correct sample
        indexInFile = index - self.fileInMemoryFirstIndex
        return_data = {}
        for key in self.data.keys():
            return_data[key] = self.data[key][indexInFile]
        return return_data

    def __len__(self):
        return sum(self.num_per_file)

class OrderedRandomSampler(object):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_per_file = self.data_source.num_per_file

    def __iter__(self):
        indices=np.array([],dtype=np.int64)
        prev_file_end = 0
        for i in range(len(self.num_per_file)):
            indices=np.append(indices, np.random.permutation(self.num_per_file[i])+prev_file_end)
            prev_file_end += self.num_per_file[i]
        return iter(from_numpy(indices))

    def __len__(self):
        return len(self.data_source)
