source /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/init_ilcsoft.sh

# generate using input generator file
#ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/lcgeo/HEAD/CLIC/compact/CLIC_o3_v07/CLIC_o3_v07.xml -O ROOTFiles/eejj.root -N 1000 --random.seed 6546777 --inputFiles GeneratorFiles/eejj.hepevt 

# particle gun input
for energy in `seq 50 10 100`
do
    for fileN in {1..10}
    do
        ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/lcgeo/HEAD/CLIC/compact/CLIC_o3_v07/CLIC_o3_v07.xml -O ROOTFiles/pi0_${energy}_GeV_${fileN}.root -N 1000 --enableGun --gun.particle pi0 --gun.energy $energy*GeV --gun.distribution uniform
        ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/lcgeo/HEAD/CLIC/compact/CLIC_o3_v07/CLIC_o3_v07.xml -O ROOTFiles/gamma_${energy}_GeV_${fileN}.root -N 1000 --enableGun --gun.particle gamma --gun.energy $energy*GeV --gun.distribution uniform
    done
done
