for i in {1..8}
do
    for j in {1..10}
    do
        python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 /data/LCD/NewSamples/RandomAngle/ATLAS/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 0
        python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 /data/LCD/NewSamples/RandomAngle/CMS/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5 1
        python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/Pi0Escan_RandomAngle_MERGED/Pi0Escan_RandomAngle_${i}_${j}.h5 /data/LCD/NewSamples/RandomAngle/ATLAS/Pi0Escan_RandomAngle_MERGED/Pi0Escan_RandomAngle_${i}_${j}.h5 0
        python resample.py /data/LCD/NewSamples/RandomAngle/CLIC/Pi0Escan_RandomAngle_MERGED/Pi0Escan_RandomAngle_${i}_${j}.h5 /data/LCD/NewSamples/RandomAngle/CMS/Pi0Escan_RandomAngle_MERGED/Pi0Escan_RandomAngle_${i}_${j}.h5 1
    done
done
