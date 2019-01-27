import glob
import torch.utils.data as data
from timeit import default_timer as timer
import blunt_loader as loader

#################################
# Load files and set up loaders #
#################################

base_path = "/public/data/calo/RandomAngle/CLIC/"
sample_path = [base_path + "Gamma/*.h5", base_path + "Pi0/*.h5"]
class_pdg_id = [22, 111]
n_classes = 2
batch_size = 1000
n_workers = 2

# gather sample files for each type of particle
class_files = [[]] * n_classes
for i, class_path in enumerate(sample_path):
    class_files[i] = glob.glob(class_path)
files_per_class = min([len(files) for files in class_files])

# split the train, test, and validation files
# get lists of [[class1_file1, class2_file1], [class1_file2, class2_file2], ...]
train_files = []
for i in range(files_per_class):
    new_files = []
    for j in range(n_classes):
        new_files.append(class_files[j][i])
    train_files.append(new_files)

# prepare the generators
print('Preparing data loaders')
train_set = loader.HDF5Dataset(train_files, class_pdg_id)
train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=loader.OrderedRandomSampler(train_set), num_workers=n_workers)


################
# Time loaders #
################

def time_loading():
    start = timer()
    for i, _ in enumerate(train_loader):
        if i > 10:
            break
    end = timer()
    print(f"Loading 10 batches took {end-start} seconds.")


#################
# Perform tests #
#################

if __name__ == "__main__":
    time_loading()
