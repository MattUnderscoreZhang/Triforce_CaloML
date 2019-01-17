#declare -a infiles=("Ele" "Gamma" "Pi0")
declare -a infiles=("Gamma" "Pi0")
for folder_count in {1..8}; do
    for file_count in {1..10}; do
        for folder_name in ${infiles[@]}; do
            echo python convert_sparse.py "/data/LCD/NewSamples/Fixed/"$folder_name"Escan_"$folder_count"_MERGED/"$folder_name"Escan_"$folder_count"_"$file_count".h5" "/data/LCD/NewSamples/Fixed/Sparse/"$folder_name"Escan_"$folder_count"_"$file_count".h5"
        done
    done
done

#declare -a infiles=("Ele" "Gamma" "Pi0" "ChPi")
#for folder_count in {1..8}; do
    #for file_count in {1..10}; do
        #for folder_name in ${infiles[@]}; do
            #python convert_sparse.py "/data/LCD/NewSamples/RandomAngle/"$folder_name"Escan_RandomAngle_MERGED/"$folder_name"Escan_"$folder_count"_"$file_count".h5" "/data/LCD/NewSamples/RandomAngle/Sparse/"$folder_name"Escan_"$folder_count"_"$file_count".h5"
        #done
    #done
#done
