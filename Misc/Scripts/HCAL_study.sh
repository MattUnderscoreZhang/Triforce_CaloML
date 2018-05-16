for iteration in {0..9}
do
    for problemType in {1..2}
    do
        for hasHCAL in {0..1}
        do
            name="NIPS_"
            if [ $problemType == 1 ]; then
                name="${name}EleChPi_"
            else
                name="${name}GammaPi0_"
            fi
            if [ $hasHCAL == 0 ]; then
                name="${name}WithHCAL_"
            else
                name="${name}NoHCAL_"
            fi
            name="${name}${iteration}"
            python3 triforce.py $name $problemType $hasHCAL
        done
    done
done
