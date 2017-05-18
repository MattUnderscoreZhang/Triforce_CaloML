#!/bin/csh
if( ! $?LD_LIBRARY_PATH ) then
  setenv LD_LIBRARY_PATH /afs/cern.ch/user/m/mazhang/Projects/JetCalo/HepMC/lib
else
  setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/afs/cern.ch/user/m/mazhang/Projects/JetCalo/HepMC/lib
endif
setenv PYTHIA8DATA ${PYTHIA8_HOME}/xmldoc
