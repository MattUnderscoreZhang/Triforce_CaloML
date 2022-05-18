source /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/init_ilcsoft.sh
 
# generate using input generator file
#ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/lcgeo/HEAD/CLIC/compact/CLIC_o3_v07/CLIC_o3_v07.xml -O ROOTFiles/eejj.root -N 1000 --random.seed 6546777 --inputFiles GeneratorFiles/eejj.hepevt 

# create output folder
mkdir ROOTFiles

# particle gun input
for energy in `seq 50 10 50`
do
    #for fileN in {1..10}
    for fileN in {1..1}
    do
        ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/lcgeo/HEAD/CLIC/compact/CLIC_o3_v07/CLIC_o3_v07.xml -O ROOTFiles/pi0_${energy}_GeV_${fileN}.root -N 1000 --enableGun --gun.particle pi0 --gun.energy $energy*GeV --gun.distribution uniform --gun.direction 1,0,0
        ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-11-22/x86_64-slc6-gcc48-opt/lcgeo/HEAD/CLIC/compact/CLIC_o3_v07/CLIC_o3_v07.xml -O ROOTFiles/gamma_${energy}_GeV_${fileN}.root -N 1000 --enableGun --gun.particle gamma --gun.energy $energy*GeV --gun.distribution uniform --gun.direction 1,0,0
    done
done







# Maurizio's code (to fix eta distribution bug)

#script.write("ddsim --compactFile /cvmfs/clicdp.cern.ch/iLCSoft/builds/2016-04-06/lcgeo/HEAD/CLIC/compact/CLIC_o2_v04/CLIC_o2_v04.xml --enableGun --gun.direction \"1.0,0.0,0.0\" --gun.particle %s%s --gun.energy %f*GeV -N 1 -O %s/%s_%i_%i.root --random.seed %i > /dev/null \n" %(args.gun, charge, energy, args.joblabel, args.joblabel, i, j, seed+111*j))
