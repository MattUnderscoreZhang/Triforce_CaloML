# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np
import ctypes
from multiprocessing import Lock, RLock, Event, Barrier, Condition, Manager
from multiprocessing.sharedctypes import  Array, RawArray, Value 
import time 
from timeit import default_timer as timer
import sys

def load_hdf5(file, pdgIDs, loadMinimalFeatures=None):
    '''Loads H5 file. Used by HDF5Dataset.'''
    return_data = {}
    with h5py.File(file, 'r') as f:
        # (default) load full ECAL / HCAL arrays and standard features
        if loadMinimalFeatures is None:
            return_data['ECAL'] = np.empty((200, 51, 51, 25), dtype='float32')
            print(f['ECAL'].shape)
            f['ECAL'].read_direct(return_data['ECAL'])
            n_events = len(return_data['ECAL'])
            return_data['HCAL'] = f['HCAL'][:].astype(np.float32)
            return_data['pdgID'] = f['pdgID'][:].astype(np.int)
            return_data['classID'] = np.array([pdgIDs[abs(i)] for i in return_data['pdgID']]) # PyTorch expects class index instead of one-hot
            
            other_features = ['ECAL_E', 'HCAL_E', 'HCAL_ECAL_ERatio', 'energy', 'eta', 'recoEta', 'phi', 'recoPhi', 'openingAngle']
            for feat in other_features:
                if feat in f.keys(): return_data[feat] = f[feat][:].astype(np.float32)
                else: return_data[feat] = np.zeros(n_events, dtype=np.float32)
        # minimal data load: only load specific features that are requested
        else:
            for feat in loadMinimalFeatures:
                return_data[feat] = f[feat][:]
    return return_data

class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Used to create PyTorch DataLoader.
    Arguments:
        dataname_tuples: list of filename tuples, where each tuple will be mixed into a single file
        num_per_file: number of events in each data file
    """
    
    def __init__(self, dataname_tuples, pdgIDs,  nWorkers, num_loaders, filters=[]):
        self.dataname_tuples = sorted(dataname_tuples)
        self.nClasses = len(dataname_tuples[0])
        self.total_files = len(dataname_tuples) # per class
        self.num_per_file = len(dataname_tuples) * [0]  
        self.num_loaders = num_loaders
        self.lock = RLock()
        self.fileInMemory = Value('i', 0, lock=self.lock) 
        self.fileInMemoryFirstIndex = Value('i', 0, lock=self.lock)
        self.fileInMemoryLastIndex = Value('i', -1, lock=self.lock)
        self.mem_index = Value('i', 1) # either 0 or 1. used for mem management.
        self.loadNext = Event()
        self.loadFile = Event()
        self.load_barrier = Barrier(self.num_loaders+1)
        self.batch_barrier = Barrier(nWorkers - (self.num_loaders+1))
        self.worker_files = [RawArray(ctypes.c_char, len(dataname_tuples[0][0])+50) for _ in range(self.num_loaders)]
        self.data = {}
        ###########################################
        # prepare memory to share with workers #
        # take a sample file and get keys and over allocate
        # we should overallocate for both classes 
        # if user runs into memory problems, use fewer num_loaders.
        with h5py.File(dataname_tuples[0][0]) as sample:
            for key in sample.keys(): 
#                 print(key)
                old_shape = sample[key].shape
                size = self.nClasses*self.num_loaders
                self.new_shape = list(old_shape)
                for dim in old_shape:
                    size *= dim
                self.new_shape[0] = self.nClasses*self.num_loaders*old_shape[0]  
                buff = RawArray(ctypes.c_float, size) # prepare mem for num_loaders
                self.data[key] = np.frombuffer(buff, dtype=np.float32).reshape(self.new_shape) # map numpy array on buffer
        ###########################################
        self.pdgIDs = {}
        self.filters = filters
        for i, ID in enumerate(pdgIDs):
            self.pdgIDs[ID] = i
        self.countEvents()
        print(self.num_per_file)
        print(len(self.num_per_file))
        

    def countEvents(self): # counter will be wrong. Need to redesign this to accomodate new files system
        # make minimal list of inputs needed to check (or count events)
        minFeatures = []
        for filt in self.filters: minFeatures += filt.featuresUsed
        # use energy to count number of events if no filters
        if len(minFeatures) == 0: minFeatures.append('energy')
        totalevents = 0
        # num_per_file and totalevents count the minimum events in a file tuple (one file for each class)
        for fileN in range(len(self.dataname_tuples)):
            nevents_before_filtering, nevents_after_filtering = [], []
            for dataname in self.dataname_tuples[fileN]:
                file_data = load_hdf5(dataname, self.pdgIDs, minFeatures)
                nevents_before_filtering.append(len(list(file_data.values())[0]))
                for filt in self.filters: filt.filter(file_data)
                nevents_after_filtering.append(len(list(file_data.values())[0]))
            self.num_per_file[fileN] = min(nevents_after_filtering) * self.nClasses
            totalevents += min(nevents_before_filtering) * self.nClasses
        print('total events:',totalevents)
        if len(self.filters) > 0:
            print('total events passing filters:',sum(self.num_per_file))

    def load_file(self, dataname, index, mem_offset): 
        if dataname != "-1".encode('utf-8'):
            file_data = load_hdf5(dataname, self.pdgIDs)
            # apply any filters here
            if not self.filters is None:
                for filt in self.filters: filt.filter(file_data)
            for key in file_data.keys():
                if key in self.data.keys(): 
                    # do something about the 200 hard coded number
                    self.data[key][200*(index-1) + mem_offset:200*index + mem_offset] = file_data[key][:] 
                    # 200 is a weak assumption.
    
    def init_worker(self, worker_id):
        # write description for function.
        if worker_id == 0: #proxy worker for old implementation. 
            while self.fileInMemory.value < self.total_files: 
                # periodically check if file needs to be loaded
                # only break when all files are loaded
                self.loadNext.wait()
                for i in range(self.nClasses):
                    if self.mem_index.value: 
                        self.mem_index.value = 0
                    else: self.mem_index.value = 1
                    # gather files for loaders
                    if self.total_files - self.fileInMemory.value < self.num_loaders:
                        file_names = [file[i] for file in self.dataname_tuples[self.fileInMemory.value:]]
                        for _ in range(self.num_loaders - (self.total_files - self.fileInMemory.value)):
                            file_names.append("-1")
                    else:
                        file_names = [file[i] for file in self.dataname_tuples[self.fileInMemory.value:self.fileInMemory.value+self.num_loaders]]
                    # share file names with loaders
                    for ind in range(len(file_names)):
                        # files to load using file loader workers. 
                        self.worker_files[ind].value = file_names[ind].encode('utf-8') # convert to bytes to store in RawArray 
                    #event to notify loader workers
                    self.loadFile.set() 
                    time.sleep(2)
                    self.loadFile.clear()
                    self.load_barrier.wait()
                    # print("process 0: escaped threading.")

                # update indicies
                with self.lock:
                    self.fileInMemoryFirstIndex.value += int(self.fileInMemoryLastIndex.value+1)
                    self.fileInMemoryLastIndex.value += sum(self.num_per_file[self.fileInMemory.value:self.fileInMemory.value+self.num_loaders])
                    self.fileInMemory.value += self.num_loaders
                
                self.loadNext.set()  
                self.loadNext.clear() # tell batch collectors that we are done. 

        if worker_id <= self.num_loaders: 
            while self.fileInMemory.value < self.total_files:
                self.loadFile.wait()
                mem_offset = self.mem_index.value * int(self.new_shape[0]/2)
                self.load_file(self.worker_files[worker_id - 1].value, worker_id, mem_offset)
                self.load_barrier.wait() # barrier object here so we can wait for all threads to finish loading
#                 print("Process %d Done"%(worker_id + mem_offset))
        # else: print("Batcher %d"%worker_id)
           
    def __getitem__(self, index):  
        # if entering a new epoch, re-initialze necessary variables
        if (index < self.fileInMemoryFirstIndex.value):
            with self.lock:
                self.fileInMemory.value = 0
                self.fileInMemoryFirstIndex.value = 0
                self.fileInMemoryLastIndex.value = -1
        # if we started to look at a new file, read the file data
        while (index > self.fileInMemoryLastIndex.value):
            self.batch_barrier.wait()
            with self.lock: 
                self.loadNext.set()
                self.loadNext.clear() # tell process 0 that we need files. 
            # wait till next file is loaded by loading_workers before threading on batch.
            self.batch_barrier.wait()
            self.loadNext.wait()
            # print("batchers escaped.")
            
        # return the correct sample
        indexInFile = index - self.fileInMemoryFirstIndex.value
#         print(indexInFile)
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
        return len(sum(self.num_per_file))
    