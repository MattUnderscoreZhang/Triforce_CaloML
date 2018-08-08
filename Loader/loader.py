import torch.utils.data as data
from torch import from_numpy
import h5py as h5
import numpy as np
import pdb
import threading

# Parts taken from https://www.sagivtech.com/2017/09/19/optimizing-pytorch-training-code/

# Makes an iterator threadproof so that multiple processes can't call it simultaneously
class ThreadsafeIter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return self.it.__next__()

# Threadproof versions of these functions are used by the loader
def get_file_index(max_index):
    current_index = 0
    while True:
        current_index = (current_index + 1) % max_index
        yield current_index
def get_event_index(max_index):
    for current_index in range(max_index):
        yield current_index

# Loader takes list of files and loops through them
class ThreadedLoader:

    def __init__(self, files, batch_size, pdgIDs, filters=[]):
        self.files = files
        self.batch_size = batch_size
        self.pdgIDs = {}
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.filters = filters
        self.filter_features = []
        for filt in self.filters:
            self.filter_features += filt.featuresUsed
        # use energy to count number of events if no filters
        if len(self.filter_features) == 0:
            self.filter_features.append('energy')
        self.file_index_iterator = ThreadsafeIter(get_file_index(len(self.files)))

    # Prepare an iterator with max_index equal to the max index of a file
    def prep_file(self, file_n):
        # self.current_file = h5.File(self.files[file_n], 'r')
        self.current_file = h5.File(self.files[file_n][0], 'r') # TEMP - ONLY FIRST CLASS
        self.event_index_iterator = ThreadsafeIter(get_event_index(len(self.current_file['ECAL'])))

    # # Event filter
    # def filter(self, event):
        # return filt.filter(event) # does this event pass

    def __iter__(self):
        for file_index in self.file_index_iterator:
            self.prep_file(file_index)
            for event_index in self.event_index_iterator:
                return_data = {}
                for feat in ['ECAL', 'HCAL', 'ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle']:
                    if feat in self.current_file.keys():
                        return_data[feat] = self.current_file[feat][event_index].astype(np.float32)
                return_data['pdgID'] = self.current_file['pdgID'][event_index].astype(int)
                return_data['classID'] = self.pdgIDs[abs(return_data['pdgID'])] # PyTorch expects class index instead of one-hot
                yield file_index, event_index

    def __next__(self):
        return self.__iter__()
