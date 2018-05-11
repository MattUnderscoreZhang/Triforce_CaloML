hl=(4 5 6)
ne=(256 512 1024)
lr=(0.001 0.005 0.01)

for hl_i in ${hl[@]}
do
    for ne_i in ${ne[@]}
    do
        for lr_i in ${lr[@]}
        do
            sed 's/HYPERPARAMETERS_SEPARATED/'${hl_i}' '${ne_i}' '${lr_i}'/g' <qsub_template.in >qsub.in
            sed 's/HYPERPARAMETERS/'${hl_i}_${ne_i}_${lr_i}'/g' -i qsub.in
            qsub qsub.in
        done
    done
done
