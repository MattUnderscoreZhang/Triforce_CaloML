for i in {1..8}
do
    for j in {1..10}
    do
        #python add_R9.py /data/LCD/NewSamples/RandomAngle/ChPiEscan_RandomAngle_MERGED/ChPiEscan_RandomAngle_${i}_${j}.h5
        #python add_R9.py /data/LCD/NewSamples/RandomAngle/EleEscan_RandomAngle_MERGED/EleEscan_RandomAngle_${i}_${j}.h5
        python add_R9.py /data/LCD/NewSamples/RandomAngle/GammaEscan_RandomAngle_MERGED/GammaEscan_RandomAngle_${i}_${j}.h5
        python add_R9.py /data/LCD/NewSamples/RandomAngle/Pi0Escan_RandomAngle_MERGED/Pi0Escan_RandomAngle_${i}_${j}.h5
    done
done
