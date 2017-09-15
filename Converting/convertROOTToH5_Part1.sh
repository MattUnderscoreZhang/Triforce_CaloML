source /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-04-06/init_ilcsoft.sh

for energy in {1..10}
do
    python Convert_to_txt.py ../AllFiles/ROOTFiles/v1/WithOpeningAngle/pi0_${energy}.root ../AllFiles/TxtFiles/v1/WithOpeningAngle/pi0_${energy}.txt
    python Convert_to_txt.py ../AllFiles/ROOTFiles/v1/WithOpeningAngle/gamma_${energy}.root ../AllFiles/TxtFiles/v1/WithOpeningAngle/gamma_${energy}.txt
done
