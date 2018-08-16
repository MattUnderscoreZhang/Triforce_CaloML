ne=(128 256 512 1024 2048 4096)
lr=(0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007)
dr=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)

sp=(1 2 3 4 5 6 7 8 9 10)

for sp_i in ${sp[@]}
do
    cp qsub_template.in qsub.in
    sed 's/JOBNAME/'${sp_i}'/g' -i qsub.in
    for ne_i in ${ne[@]}
    do
        for lr_i in ${lr[@]}
        do
            for dr_i in ${dr[@]}
            do
                echo "aprun -n 1 python3 triforce.py \"Variable/GN/GN_Output_${ne_i}_${lr_i}_${dr_i}_Scan${sp_i}\" ${ne_i} ${lr_i} ${dr_i} > Output/Variable/GN/Output_GN_${ne_i}_${lr_i}_${dr_i}_Scan${sp_i}_log.txt &" >> qsub.in
            done
        done
    done
    echo "wait" >> qsub.in
    qsub qsub.in
done
