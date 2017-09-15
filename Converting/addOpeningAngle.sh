source /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-04-06/init_ilcsoft.sh

for i in {1..10}
do
    python addOpeningAngle.py /afs/cern.ch/user/m/mazhang/Projects/CaloSampleGeneration/AllFiles/ROOTFiles/v1/pi0_60_GeV_$i.root pi0_$i.root
    python addOpeningAngle.py afs/cern.ch/user/m/mazhang/Projects/CaloSampleGeneration/AllFiles/ROOTFiles/v1/gamma_60_GeV_$i.root gamma_$i.root
done
