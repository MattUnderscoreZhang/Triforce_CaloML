hl=(3 4 5 6 7)
ne=(64 128 256 512 1024)
lr=(0.0001 0.0005 0.001)
dp=(0.2 0.3 0.4 0.5)
ws=(11 21 31 41 51)

for hl_i in ${hl[@]}
do
    for ne_i in ${ne[@]}
    do
        for lr_i in ${lr[@]}
        do
            cp qsub_template.in qsub.in
            sed 's/JOBNAME/'${hl_i}_${ne_i}_${lr_i}'/g' -i qsub.in
            for dp_i in ${dp[@]}
            do
                for ws_i in ${ws[@]}
                do
                    echo "aprun -n 1 python3 triforce.py Output_HYPERPARAMETERS HYPERPARAMETERS_SEPARATED" >> qsub.in
                    sed 's/HYPERPARAMETERS_SEPARATED/'${hl_i}' '${ne_i}' '${lr_i}' '${dp_i}' '${ws_i}'/g' -i qsub.in
                    sed 's/HYPERPARAMETERS/'${hl_i}_${ne_i}_${lr_i}_${dp_i}_${ws_i}'/g' -i qsub.in
                done
            done
            qsub qsub.in
        done
    done
done
