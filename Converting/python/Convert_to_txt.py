# Remember that we need to do this beforehand
# source /afs/cern.ch/eng/clic/work/ilcsoft/HEAD-2016-04-06/init_ilcsoft.sh

import math
import ROOT
import sys

ROOT.gSystem.Load('libDDG4')
ROOT.gSystem.Load('libDDCore')
ROOT.gSystem.Load('libDDG4Plugins')

import argparse
from pyLCIO import UTIL

def getEnergies(path_to_file, outfile, openingAngleCut):

  idDecoder = UTIL.BitField64("system:5,side:2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16")
  idDecoderHCAL = UTIL.BitField64("system:5,side:2,module:8,stave:4,layer:9,submodule:4,x:32:-16,y:-16")
  
  fil = ROOT.TFile.Open(str(path_to_file), "read")
  iEvt = 0
    
  # We make an empty list of events
  event_list = []

  for event in fil.EVENT:

    openingAngle = -1 # default if there is no pi0 -> /gamma/gamma

    if (len(event.MCParticles)>=3 and event.MCParticles[0].pdgID==111 and event.MCParticles[1].pdgID==22 and event.MCParticles[2].pdgID==22):

      gamma1px = event.MCParticles[1].psx
      gamma1py = event.MCParticles[1].psy
      gamma1pz = event.MCParticles[1].psz
      gamma2px = event.MCParticles[2].psx
      gamma2py = event.MCParticles[2].psy
      gamma2pz = event.MCParticles[2].psz
  
      cos_theta = (gamma1px*gamma2px+gamma1py*gamma2py+gamma1pz*gamma2pz)/math.sqrt((gamma1px*gamma1px+gamma1py*gamma1py+gamma1pz*gamma1pz)*(gamma2px*gamma2px+gamma2py*gamma2py+gamma2pz*gamma2pz))
      openingAngle = math.acos(cos_theta) 
  
    if (!openingAngleCut or (openingAngle < 0.01 and openingAngle != -1)): # either require opening angle cut of 0.01 or allow everything

      # We make an empty list of hits (within this event)
      hit_list = []
      hit_listHCAL = []

      iEvt = iEvt + 1

      # Read HCAL
      for i in range(len(event.HCalBarrelCollection)):
        idDecoderHCAL.setValue(event.HCalBarrelCollection[i].cellID)
  
        z = idDecoderHCAL['layer'].value()
        x = idDecoderHCAL['x'].value()
        y = idDecoderHCAL['y'].value()
        E = event.HCalBarrelCollection[i].energyDeposit
        pos = event.HCalBarrelCollection[i].position
        hit_listHCAL.append((int(x), int(y), int(z), E, pos.X(), pos.Y(), pos.Z()))

      # Read ECAL
      for i in range(len(event.ECalBarrelCollection)):

        #print event.ECalBarrelCollection.getParameters()
        idDecoder.setValue(event.ECalBarrelCollection[i].cellID)

        z = idDecoder['layer'].value()
        x = idDecoder['x'].value()
        y = idDecoder['y'].value()
        E = event.ECalBarrelCollection[i].energyDeposit
        pos = event.ECalBarrelCollection[i].position
        if (z < 25):
            hit_list.append((int(x), int(y), int(z), E, pos.X(), pos.Y(), pos.Z()))

      # Read energy
      gunpx = event.MCParticles[0].psx
      gunpy = event.MCParticles[0].psy
      gunpz = event.MCParticles[0].psz
      m = event.MCParticles[0].mass
      gunE = ROOT.TMath.Sqrt(m*m + gunpx*gunpx + gunpy*gunpy + gunpz*gunpz)
      pdgID = event.MCParticles[0].pdgID

      photon_conversion_num = 0
      for i in range(len(event.MCParticles)):
        if (event.MCParticles[i].pdgID == -11):
          positron_px = event.MCParticles[i].psx
          positron_py = event.MCParticles[i].psy
          positron_pz = event.MCParticles[i].psz
          positron_m = event.MCParticles[i].mass
          positron_E = ROOT.TMath.Sqrt(positron_m*positron_m + positron_px*positron_px + positron_py*positron_py + positron_pz*positron_pz)
          for j in range(len(event.MCParticles)):
            if (event.MCParticles[j].pdgID == 11 and event.MCParticles[j].vsx == event.MCParticles[i].vsx and event.MCParticles[j].vsy == event.MCParticles[i].vsy and event.MCParticles[j].vsz == event.MCParticles[i].vsz):
              
              electron_px = event.MCParticles[j].psx
              electron_py = event.MCParticles[j].psy
              electron_pz = event.MCParticles[j].psz
              electron_m = event.MCParticles[j].mass
              electron_E = ROOT.TMath.Sqrt(electron_m*electron_m + electron_px*electron_px + electron_py*electron_py + electron_pz*electron_pz)

              #if((positron_E + electron_E) > (ROOT.TMath.Sqrt(electron_m*electron_m+10)+ROOT.TMath.Sqrt(positron_m*positron_m +10))):
              if((positron_E + electron_E) > 1000):
                photon_conversion_num += 1
 
      event_list.append({'pdgID' : pdgID, 'E': gunE, 'px':gunpx, 'py':gunpy, 'pz':gunpz, 'conversion': photon_conversion_num, 'ECAL': hit_list, 'HCAL': hit_listHCAL, 'openingAngle' : openingAngle})

      print(len(hit_list), len(hit_listHCAL))
  
  # Append this event to the event list
  text_file = open(outfile, "w") 
  for evt in event_list:  
    text_file.write(str(evt)+"\n")
  text_file.close()

if __name__ == "__main__":
  
  inFile = sys.argv[1]
  outFile = sys.argv[2]
  openingAngleCut = sys.argv[3].astype(bool)
  getEnergies(inFile, outFile, openingAngleCut)

  # # convert the root file to txt file
  # out = sys.argv[1].replace(".root",".txt")
  # getEnergies(sys.argv[1], out)
