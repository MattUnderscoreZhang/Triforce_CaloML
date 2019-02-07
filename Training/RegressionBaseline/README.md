# LCDRegression

This repository contains some basic tools for doing energy regression studies on the simulated LCD Calorimeter datasets.  The main dependency is xgboost, which is not present by default on culture-plate at Caltech.

For all of these scripts, the input samples to use are set near the top of the file, and other options can be modified inside.

To run linear regression:
```
python train_linreg_sums.py
```

To run xgboost regression:
```
python train_xgb_features.py
```

To use a previous xgboost training to evaluate on a new dataset, modify the script below to point to the correct files and features, and run:
```
python eval_xgb_features.py <output_label>
```

To make comparison plots of energy bias and resolution, modify the script below to point to the trainings you want to compare, and run:
```
python compare_reg_plots.py
```
