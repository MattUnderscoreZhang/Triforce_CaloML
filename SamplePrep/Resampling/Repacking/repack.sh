#!/usr/bin/zsh

rm -rf Pi0 ChPi Gamma Ele
mkdir Pi0 ChPi Gamma Ele

CLIC_PATH="/public/data/calo/FixedAngle/CLIC"
SAMPLE_PREFIXES=("Gamma/GammaEscan" "Pi0/Pi0Escan")

parallel source repack_one.sh ::: ${SAMPLE_PREFIXES} ::: {1..8} ::: {1..10} ::: $CLIC_PATH
wait

rm -rf Pi0 ChPi Gamma Ele
mkdir Pi0 ChPi Gamma Ele

CLIC_PATH="/public/data/calo/RandomAngle/CLIC"
SAMPLE_PREFIXES=("Gamma/GammaEscan_RandomAngle" "Pi0/Pi0Escan_RandomAngle" "Ele/EleEscan_RandomAngle" "Pi0/Pi0Escan_RandomAngle")

parallel source repack_one.sh ::: ${SAMPLE_PREFIXES} ::: {1..8} ::: {1..10} ::: $CLIC_PATH
