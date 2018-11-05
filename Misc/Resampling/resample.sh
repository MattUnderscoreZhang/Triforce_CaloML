python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_1_1.h5 /data/LCD/NewSamples/RandomAngle/ATLAS/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_1_1.h5 0

#for i in {1..8}
#do
    #for j in {1..10}
    #do
        #python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 /data/LCD/NewSamples/RandomAngle/ATLAS/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 0
        #python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 /data/LCD/NewSamples/RandomAngle/CMS/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 1
    #done
#done
