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
            python resample.py ${CLIC_PATH}/${PREFIX}_${i}_${j}.h5 ${ATLAS_PATH}/${PREFIX}_${i}_${j}.h5 "ATLAS"
            python resample.py ${CLIC_PATH}/${PREFIX}_${i}_${j}.h5 ${CMS_PATH}/${PREFIX}_${i}_${j}.h5 "CMS"
        done
    done
done
