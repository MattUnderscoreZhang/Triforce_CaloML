#!/bin/bash

CLIC_PATH="/public/data/calo/FixedAngle/CLIC"
ATLAS_PATH="/public/data/calo/FixedAngle/ATLAS"
CMS_PATH="/public/data/calo/FixedAngle/CMS"
SAMPLE_PREFIXES=("Gamma/GammaEscan" "Pi0/Pi0Escan")

for PREFIX in $SAMPLE_PREFIXES
do
    for i in {1..8}
    do
        for j in {1..10}
        do
            sem -j 16 python resample.py ${CLIC_PATH}/${PREFIX}_${i}_${j}.h5 ${ATLAS_PATH}/${PREFIX}_${i}_${j}.h5 "ATLAS"
        done
    done
done

sem --wait

for PREFIX in $SAMPLE_PREFIXES
do
    for i in {1..8}
    do
        for j in {1..10}
        do
            sem -j 16 python resample.py ${CLIC_PATH}/${PREFIX}_${i}_${j}.h5 ${CMS_PATH}/${PREFIX}_${i}_${j}.h5 "CMS"
        done
    done
done

CLIC_PATH="/public/data/calo/RandomAngle/CLIC"
ATLAS_PATH="/public/data/calo/RandomAngle/ATLAS"
CMS_PATH="/public/data/calo/RandomAngle/CMS"
SAMPLE_PREFIXES=("Gamma/GammaEscan_RandomAngle" "Pi0/Pi0Escan_RandomAngle")

for PREFIX in $SAMPLE_PREFIXES
do
    for i in {1..8}
    do
        for j in {1..10}
        do
            sem -j 16 python resample.py ${CLIC_PATH}/${PREFIX}_${i}_${j}.h5 ${ATLAS_PATH}/${PREFIX}_${i}_${j}.h5 "ATLAS"
        done
    done
done

sem --wait

for PREFIX in $SAMPLE_PREFIXES
do
    for i in {1..8}
    do
        for j in {1..10}
        do
            sem -j 16 python resample.py ${CLIC_PATH}/${PREFIX}_${i}_${j}.h5 ${CMS_PATH}/${PREFIX}_${i}_${j}.h5 "CMS"
        done
    done
done
