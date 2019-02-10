#!/usr/bin/zsh

ATLAS_PATH="/public/data/calo/FixedAngle/ATLAS"
CMS_PATH="/public/data/calo/FixedAngle/CMS"
SAMPLE_PREFIXES=("Gamma/GammaEscan" "Pi0/Pi0Escan")

rm -rf Pi0 ChPi Gamma Ele
mkdir Pi0 ChPi Gamma Ele
parallel source add_features_cap_jobs.sh ::: ${SAMPLE_PREFIXES} ::: {1..8} ::: {1..10} ::: $ATLAS_PATH
wait

rm -rf Pi0 ChPi Gamma Ele
mkdir Pi0 ChPi Gamma Ele
parallel source add_features_cap_jobs.sh ::: ${SAMPLE_PREFIXES} ::: {1..8} ::: {1..10} ::: $CMS_PATH
wait

ATLAS_PATH="/public/data/calo/RandomAngle/ATLAS"
CMS_PATH="/public/data/calo/RandomAngle/CMS"
SAMPLE_PREFIXES=("Gamma/GammaEscan_RandomAngle" "Pi0/Pi0Escan_RandomAngle")

rm -rf Pi0 ChPi Gamma Ele
mkdir Pi0 ChPi Gamma Ele
parallel source add_features_cap_jobs.sh ::: ${SAMPLE_PREFIXES} ::: {1..8} ::: {1..10} ::: $ATLAS_PATH
wait

rm -rf Pi0 ChPi Gamma Ele
mkdir Pi0 ChPi Gamma Ele
parallel source add_features_cap_jobs.sh ::: ${SAMPLE_PREFIXES} ::: {1..8} ::: {1..10} ::: $CMS_PATH
