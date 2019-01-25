# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import pdb

import h5py as h5
import numpy as np
from torch import from_numpy
import torch.utils.data as data

from Loader.filters import get_events_passing_filters, take_passing_events


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


def load_min_feature_data_and_find_passing_events(filename, filters, pdgIDs):
    def get_min_filter_features(filters):
        # make minimal list of inputs needed to check (or count events)
        minFeatures = []
        for filt in filters:
            minFeatures += filt.featuresUsed
        # use energy to count number of events if no filters
        if len(filters) == 0:
            minFeatures.append('energy')
        return minFeatures
    minFeatures = get_min_filter_features(filters)
    file_data = load_hdf5(filename, pdgIDs, minFeatures)
    passing_events = get_events_passing_filters(file_data, filters)
    return passing_events


def get_filtered_data(filename, filters, pdgIDs, passing_events):
    file_data = load_hdf5(filename, pdgIDs)
    return take_passing_events(file_data, passing_events)


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
        self.tuple_in_memory = 0
        self.tuple_in_memory_first_index = 0
        self.tuple_in_memory_last_index = -1
        self.tuple_in_memory_good_events =[]
        self.pdgIDs = {}
        self.filters = filters
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.n_events_in_file_tuple = self.count_events(self.filename_tuples, self.filters, self.pdgIDs, self.nClasses)

    def __getitem__(self, index):
        if (index < self.tuple_in_memory_first_index):  # starting new epoch
            self.prep_first_tuple()
        if(index > self.tuple_in_memory_last_index):  # finished reading this tuple 
            self.prep_next_tuple()
        index_in_tuple = index - self.tuple_in_memory_first_index
        return self.read_event_from_tuple_in_memory(index_in_tuple)

    def find_good_events_for_file_tuple(self, self.filename_tuples[self.tuple_in_memory]):
        self.tuple_in_memory_good_events = load_min_feature_data_and_find_passing_events(filename, filters, pdgIDs)

    def count_events(self, filename_tuples, filters, pdgIDs, nClasses):
        def n_filtered_events(filename):
            passing_events = load_min_feature_data_and_find_passing_events(filename, filters, pdgIDs)
            return len(passing_events)

        def min_filtered_events(filename_tuple):
            return min([n_filtered_events(filename) for filename in filename_tuple])

        n_events_in_file_tuple = [min_filtered_events(filename_tuple) * nClasses for filename_tuple in filename_tuples]
        print('total events passing filters:', sum(n_events_in_file_tuple))
        return n_events_in_file_tuple

    def prep_first_tuple(self):
        self.tuple_in_memory = 0
        self.tuple_in_memory_first_index = 0
        self.tuple_in_memory_last_index = -1

    def prep_next_tuple(self):
        self.tuple_in_memory += 1
        self.tuple_in_memory_first_index = self.tuple_in_memory_last_index + 1
        self.tuple_in_memory_last_index += self.n_events_in_file_tuple[self.tuple_in_memory]
        # print(index, self.tuple_in_memory, self.tuple_in_memory_first_index, self.tuple_in_memory_last_index)

    def read_event_from_tuple_in_memory(self, index):

        def get_filtered_data_tuple(filename_tuple, filters, pdgIDs):
            def passing_events(filename):
                load_min_feature_data_and_find_passing_events(filename, filters, pdgIDs)

            def combine_dictionaries(dictionaries):
                combined_dictionary = {}
                for dictionary in dictionaries:
                    for key in dictionary.keys():
                        if key in combined_dictionary.keys():
                            combined_dictionary[key] = np.append(combined_dictionary[key], dictionary[key], axis=0)
                        else:
                            combined_dictionary[key] = dictionary[key]
                return combined_dictionary

            file_tuple_data = [get_filtered_data(filename, filters, pdgIDs, passing_events(filename)) for filename in filename_tuple]

            return combine_dictionaries(file_tuple_data)

        data = get_filtered_data_tuple(self.filename_tuples[self.tuple_in_memory], self.filters, self.pdgIDs)

        return_data = {}
        for key in data.keys():
            return_data[key] = data[key][index]

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
