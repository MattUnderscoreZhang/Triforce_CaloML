# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np

def load_hdf5(file, pdgIDs):
    '''Loads H5 file. Used by HDF5Dataset.'''
    return_data = {}
    with h5py.File(file, 'r') as f:
        return_data['ECAL'] = f['ECAL'][:].astype(np.float32)
        n_events = len(return_data['ECAL'])
        return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
        return_data['pdgID'] = f['pdgID'][:].astype(int)
        return_data['pdgID'] = [pdgIDs[abs(i)] for i in return_data['pdgID']] # PyTorch expects class index instead of one-hot
        if 'energy' in f.keys():
            return_data['energy'] = f['energy'][:].astype(np.float32)
        else: return_data['energy'] = np.zeros(n_events, dtype=np.float32)
        if 'eta' in f.keys():
            return_data['eta'] = f['eta'][:].astype(np.float32)
        else: return_data['eta'] = np.zeros(n_events, dtype=np.float32)
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

    def __init__(self, dataname_tuples, num_per_file, pdgIDs):
        self.dataname_tuples = sorted(dataname_tuples)
        self.num_per_file = num_per_file
        self.fileInMemory = -1
        self.data = {}
        self.pdgIDs = {}
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i

    def __getitem__(self, index):
        fileN = index//self.num_per_file
        indexInFile = index%self.num_per_file-1
        # if we started to look at a new file, read the file data
        if(fileN != self.fileInMemory):
            self.data = {}
            for dataname in self.dataname_tuples[fileN]:
                file_data = load_hdf5(dataname, self.pdgIDs)
                for key in file_data.keys():
                    if key in self.data.keys():
                        self.data[key] = np.append(self.data[key], file_data[key], axis=0)
                    else:
                        self.data[key] = file_data[key]
            self.fileInMemory = fileN
        # return the correct sample
        return_data = {}
        for key in self.data.keys():
            return_data[key] = self.data[key][indexInFile]
        return return_data

    def __len__(self):
        return len(self.dataname_tuples)*self.num_per_file

class OrderedRandomSampler(object):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_per_file = self.data_source.num_per_file
        self.num_of_files = len(self.data_source.dataname_tuples)

    def __iter__(self):
        indices=np.array([],dtype=np.int64)
        for i in range(self.num_of_files):
            indices=np.append(indices, np.random.permutation(self.num_per_file)+i*self.num_per_file)
        return iter(from_numpy(indices))

    def __len__(self):
        return len(self.data_source)
