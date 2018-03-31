import os, sys

options = {'outPath': os.getcwd()+"/Output/"+sys.argv[1]+"/"}

import Tools.Classification_Plotter as Classification_Plotter

Classification_Plotter.make_all(options['outPath']+'results.h5')
