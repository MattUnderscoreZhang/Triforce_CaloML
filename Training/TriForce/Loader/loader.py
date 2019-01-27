# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import h5py as h5
import numpy as np
from torch import from_numpy
import torch.utils.data as data


def load_hdf5(file, pdgIDs, loadMinimalFeatures=None):
    '''Loads H5 file. Used by HDF5Dataset.'''
    return_data = {}
    with h5.File(file, 'r') as f:
        # (default) load full ECAL / HCAL arrays and standard features
        if loadMinimalFeatures is None:
            return_data['ECAL'] = f['ECAL'][:].astype(np.float32)
            n_events = len(return_data['ECAL'])
            return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
            return_data['pdgID'] = f['pdgID'][:].astype(int)
            return_data['classID'] = np.array([pdgIDs[abs(i)] for i in return_data['pdgID']])  # PyTorch expects class index instead of one-hot
            other_features = ['ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle']
            for feat in other_features:
                if feat in f.keys():
                    return_data[feat] = f[feat][:].astype(np.float32)
                else:
                    return_data[feat] = np.zeros(n_events, dtype=np.float32)
        # minimal data load: only load specific features that are requested
        else:
            for feat in loadMinimalFeatures:
                return_data[feat] = f[feat][:]
    return return_data


def get_min_filter_features(filters):
    # make minimal list of inputs needed to check (or count events)
    minFeatures = []
    for filt in filters:
        minFeatures += filt.featuresUsed
    # use energy to count number of events if no filters
    if len(filters) == 0:
        minFeatures.append('energy')
    return minFeatures


def get_min_filtered_data(filename, filters, pdgIDs):
    minFeatures = get_min_filter_features(filters)
    file_data = load_hdf5(filename, pdgIDs, minFeatures)
    for filt in filters:
        filt.filter(file_data)
    return file_data


def get_filtered_data(filename_tuple, filters, pdgIDs):
    data = {}
    for dataname in filename_tuple:
        file_data = load_hdf5(dataname, pdgIDs)
        for filt in filters:
            filt.filter(file_data)
        for key in file_data.keys():
            if key in data.keys():
                data[key] = np.append(data[key], file_data[key], axis=0)
            else:
                data[key] = file_data[key]
    return data


def count_events(filename_tuples, filters, pdgIDs, nClasses):
    # n_events_in_file_tuple counts the minimum events in a file tuple (one file for each class)
    n_events_in_file_tuple = len(filename_tuples) * [0]

    for tuple_n, filename_tuple in enumerate(filename_tuples):
        nevents_after_filtering = []
        for filename in filename_tuple:
            filtered_data = get_min_filtered_data(filename, filters, pdgIDs)
            nevents_after_filtering.append(len(list(filtered_data.values())[0]))
        n_events_in_file_tuple[tuple_n] = min(nevents_after_filtering) * nClasses

    print('total events passing filters:', sum(n_events_in_file_tuple))
    return n_events_in_file_tuple


class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Used to create PyTorch DataLoader.
    Arguments:
        filename_tuples: list of filename tuples, where each tuple will be mixed into a single file
        n_events_in_file_tuple: number of events in each data file
    """

    def __init__(self, filename_tuples, pdgIDs, filters=[]):
        self.filename_tuples = sorted(filename_tuples)
        self.nClasses = len(filename_tuples[0])
        self.fileInMemory = -1
        self.fileInMemoryFirstIndex = 0
        self.fileInMemoryLastIndex = -1
        self.data = {}
        self.pdgIDs = {}
        self.filters = filters
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.n_events_in_file_tuple = count_events(self.filename_tuples, self.filters, self.pdgIDs, self.nClasses)

    def __getitem__(self, index):
        # if entering a new epoch, re-initialze necessary variables
        if (index < self.fileInMemoryFirstIndex):
            self.prep_first_file()
        # if we started to look at a new file, read the file data
        if(index > self.fileInMemoryLastIndex):
            self.prep_next_file()
        # return the correct sample
        indexInFile = index - self.fileInMemoryFirstIndex
        return_data = {}
        for key in self.data.keys():
            return_data[key] = self.data[key][indexInFile]
        return return_data

    def prep_first_file(self):
        self.fileInMemory = -1
        self.fileInMemoryFirstIndex = 0
        self.fileInMemoryLastIndex = -1

    def prep_next_file(self):
        # update indices to new file
        self.fileInMemory += 1
        self.fileInMemoryFirstIndex = int(self.fileInMemoryLastIndex+1)
        self.fileInMemoryLastIndex += self.n_events_in_file_tuple[self.fileInMemory]
        # print(index, self.fileInMemory, self.fileInMemoryFirstIndex, self.fileInMemoryLastIndex)
        self.data = get_filtered_data(self.filename_tuples[self.fileInMemory], self.filters, self.pdgIDs)

    def __len__(self):
        return sum(self.n_events_in_file_tuple)


class OrderedRandomSampler(data.Sampler):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
        Note: argument 'data.Sampler' for PyTorch-0.4.1 / 'object' for PyTorch-0.3.1
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.n_events_in_file_tuple = data_source.n_events_in_file_tuple

    def __iter__(self):
        indices = np.array([], dtype=np.int64)
        prev_file_end = 0
        for i in range(len(self.n_events_in_file_tuple)):
            indices = np.append(indices, np.random.permutation(self.n_events_in_file_tuple[i])+prev_file_end)
            prev_file_end += self.n_events_in_file_tuple[i]
        return iter(from_numpy(indices))

    def __len__(self):
        return len(sum(self.n_events_in_file_tuple))
