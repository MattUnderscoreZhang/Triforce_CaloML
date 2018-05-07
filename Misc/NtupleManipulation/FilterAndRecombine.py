# takes a folder full of samples, applies an event filter, and recombines the samples

import glob
import h5py as h5

in_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/Pi0Escan_*_MERGED/Pi0Escan_*.h5"
out_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed_Filtered/Pi0Escan"
target_events_per_file = 10000

##########
# FILTER #
##########

def filter(file):

    n_events = file['ECAL'].shape[0]
    filtered_events = []

    for n in range(n_events):
        if file['HCAL_ECAL_ERatio'][n] <= 0.025:
            filtered_events.append(n)

    return filtered_events

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
    file = h5.File(file_name)
    file_keys = list(file.keys())
    new_filtered_events = filter(file)

    # add filtered events to writeout dictionary
    if starting_new_file:
        starting_new_file = False
        if n_leftover_filtered_events > 0:
            filtered_keys = leftover_filtered_keys
        else:
            for key in file_keys:
                filtered_keys[key] = []
    for key in file_keys:
        filtered_keys[key] += list(file[key][new_filtered_events])
    file.close()

    # if we've loaded enough events, write out a file
    n_new_filtered_events = len(new_filtered_events)
    n_filtered_events += n_new_filtered_events
    if n_filtered_events >= target_events_per_file:
        out_file = h5.File(out_path + "_" + str(out_file_count) + ".h5")
        n_leftover_filtered_events = n_filtered_events - target_events_per_file
        n_filtered_events = 0
        for key in file_keys:
            out_file[key] = filtered_keys[key][:target_events_per_file]
            leftover_filtered_keys[key] = filtered_keys[key][target_events_per_file:]
        out_file.close()
        out_file_count++
        starting_new_file = True

# write out any extra events
if n_leftover_filtered_events > 0:
    out_file = h5.File(out_path + "_" + str(out_file_count) + "_incomplete.h5")
    for key in file_keys:
        out_file[key] = leftover_filtered_keys[key]
    out_file.close()
