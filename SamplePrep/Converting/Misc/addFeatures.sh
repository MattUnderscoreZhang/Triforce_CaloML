for i in {0..29}
do
    python python/addFeatures.py /data/LCD/V4/DownsampledEleChPiMergingSize1Float/ChPiEscan/ChPiEscan_$i.h5 /data/LCD/V4/DownsampledEleChPiMergingSize1WithFeatures/ChPiEscan_$i.h5
    python python/addFeatures.py /data/LCD/V4/DownsampledEleChPiMergingSize1Float/EleEscan/EleEscan_$i.h5 /data/LCD/V4/DownsampledEleChPiMergingSize1WithFeatures/EleEscan_$i.h5
    python python/addFeatures.py /data/LCD/V4/DownsampledGammaPi0MergingSize1Float/Pi0Escan/Pi0Escan_$i.h5 /data/LCD/V4/DownsampledGammaPi0MergingSize1WithFeatures/Pi0Escan_$i.h5
    python python/addFeatures.py /data/LCD/V4/DownsampledGammaPi0MergingSize1Float/GammaEscan/GammaEscan_$i.h5 /data/LCD/V4/DownsampledGammaPi0MergingSize1WithFeatures/GammaEscan_$i.h5
done
