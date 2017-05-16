source /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-04-06/init_ilcsoft.sh

for energy in {9..10}
do
    python Convert_to_txt.py ../ROOTFiles/pi0_${energy}_GeV.root ../TxtFiles/pi0_${energy}_GeV.txt
    python Convert_to_txt.py ../ROOTFiles/gamma_${energy}_GeV.root ../TxtFiles/gamma_${energy}_GeV.txt
done
