# takes a folder full of samples, applies an event filter, and recombines the samples

import glob
import h5py as h5
import numpy as np

# in_path = "/data/LCD/NewSamples/Fixed/ChPiEscan*/ChPiEscan*.h5"
# out_path = "/data/LCD/NewSamples/FixedFiltered/ChPiEscan/ChPiEscan"
in_path = "/data/LCD/NewSamples/Fixed/EleEscan*/EleEscan*.h5"
out_path = "/data/LCD/NewSamples/FixedFiltered/EleEscan/EleEscan"
# in_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/Incomplete/GammaEscan_*.h5"
# out_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed_Filtered/GammaEscan"
target_events_per_file = 500

##########
# FILTER #
##########

def filter(file):

    return list(np.where(file['HCAL_ECAL_ERatio'][:] <= 0.1)[0])

###########
# COMBINE #
###########

n_filtered_events = 0
filtered_keys = {}
n_leftover_filtered_events = 0
leftover_filtered_keys = {}
out_file_count = 1
starting_new_file = True

in_files = glob.glob(in_path)
for file_name in in_files:

    # load and filter the next file
    print("Loading file", file_name)
    file = h5.File(file_name, 'r')
    file_keys = list(file.keys())
    new_filtered_events = filter(file)
    n_new_filtered_events = len(new_filtered_events)
    print(n_new_filtered_events, "events after filtering")

    # add filtered events to writeout dictionary
    print("Copying data structures (takes a long time)")
    if starting_new_file:
        starting_new_file = False
        if n_leftover_filtered_events > 0:
            filtered_keys = leftover_filtered_keys
            n_filtered_events = n_leftover_filtered_events
        else:
            for key in file_keys:
                filtered_keys[key] = []
    for key in file_keys:
        filtered_keys[key] += list(file[key][new_filtered_events])
    file.close()

    # if we've loaded enough events, write out a file
    n_filtered_events += n_new_filtered_events
    while n_filtered_events >= target_events_per_file:
        print("Writing out file", out_path + "_" + str(out_file_count) + ".h5")
        out_file = h5.File(out_path + "_" + str(out_file_count) + ".h5", 'w')
        n_filtered_events -= target_events_per_file
        if n_filtered_events < target_events_per_file:
            n_leftover_filtered_events = n_filtered_events
            n_filtered_events = 0
        for key in file_keys:
            out_file[key] = filtered_keys[key][:target_events_per_file]
            leftover_filtered_keys[key] = filtered_keys[key][target_events_per_file:]
        out_file.close()
        out_file_count += 1
        starting_new_file = True

# write out any extra events
if n_leftover_filtered_events > 0:
    print("Writing out file", out_path + "_" + str(out_file_count) + ".h5_incomplete")
    out_file = h5.File(out_path + "_" + str(out_file_count) + ".h5_incomplete", 'w')
    for key in file_keys:
        out_file[key] = leftover_filtered_keys[key]
    out_file.close()
