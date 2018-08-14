# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch
import h5py as h5
import numpy as np
import threading, queue
import pdb

class HDF5Dataset():

    def __init__(self, filename_tuples, pdgIDs, batch_size, n_workers, filters=[]):
        self.filename_tuples = sorted(filename_tuples)
        self.nClasses = len(filename_tuples[0])
        self.pdgIDs = {}
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.filters = filters
        self.files_queue = queue.Queue()
        for fileN in range(len(self.filename_tuples)):
            for filename in self.filename_tuples[fileN]:
                self.files_queue.put(filename)
        self.data_threads = []
        for _ in range(self.n_workers):
            t = threading.Thread(target=self.add_data, args=())
            t.start()
            self.data_threads.append(t)
        self.data_queue = queue.Queue(maxsize=20)

    def load_hdf5(self, filename, pdgIDs):
        return_data = {}
        with h5.File(filename, 'r') as f:
            return_data['ECAL'] = f['ECAL'][:].astype(np.float32)
            n_events = len(return_data['ECAL'])
            return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
            return_data['pdgID'] = f['pdgID'][:].astype(int)
            return_data['classID'] = np.array([pdgIDs[abs(i)] for i in return_data['pdgID']]) # PyTorch expects class index instead of one-hot
            other_features = ['ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle']
            for feat in other_features:
                if feat in f.keys(): return_data[feat] = f[feat][:].astype(np.float32)
                else: return_data[feat] = np.zeros(n_events, dtype=np.float32)
        return return_data

    def add_data(self):
        while True:
            filename = self.files_queue.get()
            self.files_queue.task_done()
            print("Loading file", filename)
            file_data = self.load_hdf5(filename, self.pdgIDs)
            print("Finished loading file", filename)
            for filt in self.filters:
                filt.filter(file_data)
            for i in range(0, file_data['ECAL'].shape[0]-self.batch_size, self.batch_size):
                data = {}
                for key in file_data.keys():
                    data[key] = torch.from_numpy(file_data[key][i:i+self.batch_size])
                print("Adding data to queue")
                self.data_queue.put(data)
                print("Done adding data to queue")
            print("Finished processing file", filename)
            self.files_queue.put(filename)

    def __next__(self):
        print("Queue size is currently", self.data_queue.qsize())
        data = self.data_queue.get()
        self.data_queue.task_done()
        return data

    def __iter__(self):
        return self
