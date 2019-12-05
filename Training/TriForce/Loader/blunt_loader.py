import sys

import h5py as h5
import numpy as np
import torch.utils.data as data

# from Loader.filters import does_event_pass_filters


class HDF5Dataset(data.Dataset):
    '''Creates a dataset from a set of H5 files. Used to create PyTorch DataLoader.'''
    def __init__(self, filename_tuples, pdgIDs, filters=[]):
        self.files, self.file_classes, self.file_nevents = self.load_files(filename_tuples)
        self.file_cumulative_nevents = np.cumsum(self.file_nevents)
        self.pdgID_to_class = {}
        for class_n, pdgID in enumerate(pdgIDs):
            self.pdgID_to_class[pdgID] = class_n
        self.filters = filters
        self.total_events = sum(self.file_nevents)

    def load_files(self, filename_tuples):
        '''Input: list of N-element tuples e.g. [(Pi0_0, Gamma_0), (Pi0_1, Gamma_1)...].
        Output: list of files, classes, and n_events.'''
        filename_lists_by_class = list(map(list, zip(*(map(list, filename_tuples)))))
        files = []
        file_classes = []
        for class_n, filename_list in enumerate(filename_lists_by_class):
            files += [h5.File(filename) for filename in filename_list]
            file_classes += [class_n for filename in filename_list]
        file_nevents = [i['ECAL'].shape[0] for i in files]
        return files, file_classes, file_nevents

    def convert_index_to_file_event_n(self, index):
        '''Converts event index to file_n, event_n.'''
        for file_n in range(len(self.file_cumulative_nevents)):
            if index < self.file_cumulative_nevents[file_n]:
                if file_n == 0:
                    return file_n, index
                else:
                    return file_n, index - self.file_cumulative_nevents[file_n-1]
        print("Index greater than total loaded events!")
        sys.exit(0)

    def __getitem__(self, index):
        file_n, event_n = self.convert_index_to_file_event_n(index)
        this_file = self.files[file_n]
        event = {}
        features = ['ECAL', 'HCAL', 'ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle', 'pdgID']
        for feat in features:
            if feat in this_file.keys():
                event[feat] = this_file[feat][event_n].astype(np.float32)
            else:
                event[feat] = np.float32(0)
                if feat == 'ECAL':
                    event[feat] = np.zeros((25, 25, 25), dtype=np.float32)
                if feat == 'HCAL':
                    event[feat] = np.zeros((11, 11, 60), dtype=np.float32)
            if feat == 'pdgID':
                pdgID = event['pdgID'].astype(int)
                event['pdgID'] = pdgID
                event['classID'] = self.pdgID_to_class[abs(pdgID)]
        return event

    def __len__(self):
        return self.total_events


class OrderedRandomSampler(data.Sampler):
    """Samples a Dataset randomly, without replacement."""
    def __init__(self, dataset):
        self.total_events = dataset.total_events

    def __iter__(self):
        return iter(np.random.permutation(self.total_events))

    def __len__(self):
        return self.total_events
