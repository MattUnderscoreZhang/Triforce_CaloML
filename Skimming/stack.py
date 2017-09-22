# Combine multiple h5 files into a single file
# Author: W. Wei
# python stack.py <Path1> <Path2>
# Path1 is a file search string, such as <Path>/Pi0*.h5
# Path2 is output files, with names <Path2>_0.h5 etc.

import h5py
import numpy as np
import sys
import glob

Path1=sys.argv[1]
Path2=sys.argv[2]

Files=glob.glob(Path1)

Keys=[]

MAX_EVENTS=10000

FilePointer=0
FileCounter=0
EventPointer=0
EventCounter=0

while(FilePointer<len(Files)):
    outfile=h5py.File(Path2+"_"+str(FileCounter)+".h5",'w')
    for i in range(FilePointer, len(Files)):
        try:
            infile=h5py.File(Files[i],'r')
            #print "Processing "+Files[i]
        except IOError:
            print "Fail to open "+Files[i]
        else:
            # recursively list all datasets in sample subgroups
            # visits each group, subgroup, and dataset in the file and adds unique dataset names to a list
            datasets = []
            def appendName(name):
                if isinstance(infile[name], h5py.Dataset):
                    if name not in datasets: datasets.append(name)
                return None
            infile.visit(appendName)

            #print "(", FileCounter, FilePointer, EventCounter, EventPointer, ") ",
            if(EventCounter<MAX_EVENTS and EventCounter+infile['ECAL/ECAL'].shape[0]-EventPointer<=MAX_EVENTS):
                for key in datasets:
                    if key in Keys:
                        OldDataset=infile[key][EventPointer:]
                        exec(key+"=np.concatenate(("+key+",OldDataset),axis=0)")
                    else:
                        OldDataset=infile[key][EventPointer:]
                        Keys.append(key)
                        exec(key+"=OldDataset[:]")
                EventCounter+=infile['ECAL'].shape[0]-EventPointer
                infile.close()
    
                if(EventPointer):
                    EventPointer=0
                
                if(EventCounter==MAX_EVENTS):
                    for key in Keys:
                        exec("outfile.create_dataset(key, data="+key+")")
    
                    outfile.close()
                    FileCounter+=1
                    EventCounter=0
                    FilePointer=i+1
                    Keys=[]
                    break
    
            elif(EventCounter<MAX_EVENTS and EventCounter+infile['ECAL/ECAL'].shape[0]-EventPointer>MAX_EVENTS):
                EventPointerPre=EventPointer
                EventPointer=MAX_EVENTS-EventCounter+EventPointerPre
                for key in datasets:
                    if key in Keys:
                        OldDataset=infile[key][EventPointerPre:EventPointer]
                        exec(key+"=np.concatenate(("+key+",OldDataset),axis=0)")
                    else:
                        OldDataset=infile[key][EventPointerPre:EventPointer]
                        Keys.append(key)
                        exec(key+"=OldDataset[:]")
                EventCounter=MAX_EVENTS
                infile.close()
                for key in Keys:
                    exec("outfile.create_dataset(key, data="+key+")")
    
                outfile.close()
                FileCounter+=1
                EventCounter=0
                FilePointer=i
                Keys=[]
                break

    if (EventCounter):
        print EventCounter, " events remaining, not written."
        break
