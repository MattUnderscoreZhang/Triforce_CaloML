echo Working on sample $1_$2_$3.h5 &&
#h5repack -l ECAL:CHUNK=1x51x51x25 -l HCAL:CHUNK=1x11x11x60 $4/$1_$2_$3.h5 $1_$2_$3.h5 &&
h5repack -l HCAL:CHUNK=1x11x11x60 $4/$1_$2_$3.h5 $1_$2_$3.h5 &&
mv -f $1_$2_$3.h5 $4/$1_$2_$3.h5
