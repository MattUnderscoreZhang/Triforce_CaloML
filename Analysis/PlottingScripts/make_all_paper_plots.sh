## 3D ECAL comparison plots between fixed-angle and random-angle pi0 events
#python PaperPlots/plotECAL.py /public/data/calo/FixedAngle/CLIC/Pi0/Pi0Escan_1_1.h5 ../Plots/CompareECAL/Fixed/CLIC/ 5
#python PaperPlots/plotECAL.py /public/data/calo/FixedAngle/ATLAS/Pi0/Pi0Escan_1_1.h5 ../Plots/CompareECAL/Fixed/ATLAS/ 5
#python PaperPlots/plotECAL.py /public/data/calo/FixedAngle/CMS/Pi0/Pi0Escan_1_1.h5 ../Plots/CompareECAL/Fixed/CMS/ 5
#python PaperPlots/plotECAL.py /public/data/calo/RandomAngle/CLIC/Pi0/Pi0Escan_RandomAngle_1_1.h5 ../Plots/CompareECAL/Random/CLIC/ 5
#python PaperPlots/plotECAL.py /public/data/calo/RandomAngle/ATLAS/Pi0/Pi0Escan_RandomAngle_1_1.h5 ../Plots/CompareECAL/Random/ATLAS/ 5
#python PaperPlots/plotECAL.py /public/data/calo/RandomAngle/CMS/Pi0/Pi0Escan_RandomAngle_1_1.h5 ../Plots/CompareECAL/Random/CMS/ 5


# make classification comparison ROC curves
python PaperPlots/FinalClassificationPlots.py
