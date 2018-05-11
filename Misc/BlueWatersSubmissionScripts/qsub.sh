#md=(5 6 7)
#ne=(400 600 800)
#lr=(0.5)
md=(5)
ne=(400)
lr=(0.5)

for md_i in ${md[@]}
do
    for ne_i in ${ne[@]}
    do
        for lr_i in ${lr[@]}
        do
            sed 's/HYPERPARAMETERS/'${md_i}_${ne_i}_${lr_i}'/g' <qsub_template.in >qsub.in
            qsub qsub.in
        done
    done
done
