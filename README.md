# Kaggle S4:E3 Competition (Steel Defect Detection)










## Files

**EDA.ipynb**

This is a notebook where I perform basic exploratory data analysis on the dataset.

**BaseScores.ipynb**

Here, I analyze the performance of XGBoost with 5-fold CV before doing any feature engineering. I also gave a baseline using one-versus-rest binary cross entropy with an additional eighth class for `No Defect`. For consistency and generalizability, I excluded `No Defect` score from mean AUROC calculation. The latter approach worked slightly better.

**FeatureGenerating.ipynb**

We create many features and observe feature importances for XGB and RF models.

**engineering.py**

This is a module that holds our feature generating function. To make optimization easier, there's an optional input for columns to drop. 

**RFOptimizing.ipynb**
**XGBOptimizing.ipynb**
**EnsembleOptimizing.ipynb**
**Submission.ipynb**