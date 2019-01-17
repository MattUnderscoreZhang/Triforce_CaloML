source /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-04-06/init_ilcsoft.sh

for n in {1..10}
do
    python python/Convert_to_txt.py ../AllFiles/ROOTFiles/v1/pi0_60_GeV_$n.root ../AllFiles/TxtFiles/v1/WithOpeningAngle/pi0_60_GeV_$n.txt true
    #python python/Convert_to_txt.py ../AllFiles/ROOTFiles/v1/gamma_60_GeV_$n.root ../AllFiles/TxtFiles/v1/WithOpeningAngle/gamma_60_GeV_$n.txt false
done
