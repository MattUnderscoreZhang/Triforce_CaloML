import glob
import h5py as h5
import numpy as np
from os import listdir
from os.path import isfile, join
import sys, getopt


helpline = 'skimH5.py -i <in_path> -o <out_path> -n <n_events_per_file>'


def filter(file):
    # return passing indices
    return [index for index, (ratio, ecal_e) in enumerate(zip(file['HCAL_ECAL_ERatio'], file["ECAL_E"])) if ratio < 0.025 and ecal_e != 0] # gamma pi0
    # return list(np.where(np.logical_and(file['energy'][:] >= 400, file['energy'][:] <= 500))[0])


if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, "hi:o:n:", ["inputPath=", "outputPath=", "nEventsPerFile="])
    except getopt.GetoptError:
        print helpline
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpline
            sys.exit()
        elif opt in ("-i", "--inputPath"):
            in_path = arg
        elif opt in ("-o", "--outputPath"):
            out_path = arg
        elif opt in ("-n", "--nEventsPerFile"):
            n_events_per_file = arg

    print 'Converting all files found in ', in_path
    print 'Saving converted files in ', out_path

    n_filtered_events = 0
    filtered_keys = {}
    out_file_count = 1

    # loop over all the files in the in_path folder
    for file_name in [file for file in listdir(in_path) if isfile(join(in_path, file))]:

        print("Loading file", file_name)
        file = h5.File(in_path + file_name, "r")
        file_keys = list(file.keys())
        new_filtered_events = filter(file)
        n_new_filtered_events = len(new_filtered_events)
        print(n_new_filtered_events, "events after filtering")

        # add filtered events to writeout dictionary
        print("Copying data structures (takes a long time)")
        n_filtered_events += n_new_filtered_events
        for key in file_keys:
            if key not in filtered_keys.keys():
                filtered_keys[key] = []
            filtered_keys[key] += list(file[key][new_filtered_events])
        file.close()

        # if we've loaded enough events, write out a file
        while n_filtered_events >= n_events_per_file:
            print("Writing out file", out_path + "_" + str(out_file_count) + ".h5")
            out_file = h5.File(out_path + "_" + str(out_file_count) + ".h5", 'w')
            for key in file_keys:
                out_file[key] = filtered_keys[key][:n_events_per_file]
            out_file.close()
            out_file_count += 1

            # remove written events from memory
            n_filtered_events -= n_events_per_file
            for key in file_keys:
                filtered_keys[key] = filtered_keys[key][n_events_per_file:]

    # write out any extra events
    if n_filtered_events > 0:
        print("Writing out file", out_path + "skimmed_" + str(out_file_count) + ".h5_incomplete")
        out_file = h5.File(out_path + "skimmed_" + str(out_file_count) + ".h5_incomplete", 'w')
        for key in file_keys:
            out_file[key] = filtered_keys[key]
        out_file.close()
