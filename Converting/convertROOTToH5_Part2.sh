#for n in {1..10}
for n in {3..10}
do
    python Convert_to_h5.py ../AllFiles/TxtFiles/v1/WithOpeningAngle/pi0_60_GeV_$n.txt ../AllFiles/H5Files/v1/Unskimmed_withOpeningAngle/pi0_60_GeV_$n.h5
    #python Convert_to_h5.py ../AllFiles/TxtFiles/v1/WithOpeningAngle/gamma_60_GeV_$n.txt ../AllFiles/H5Files/v1/Unskimmed_withOpeningAngle/gamma_60_GeV_$n.h5
done
