echo Working on sample $4/$1_$2_$3.h5 &&
python addFeatures.py $4/$1_$2_$3.h5 $1_$2_$3.h5 &&
mv -f $1_$2_$3.h5 $4/$1_$2_$3.h5
