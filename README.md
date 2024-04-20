# Kaggle S4:E3 Competition (Steel Defect Detection)



## Files

**EDA.ipynb**

This is a notebook where I perform basic exploratory data analysis on the dataset.

**BaseScores.ipynb**

Here, I analyze the performance of XGBoost with 5-fold CV before doing any feature engineering. I also gave a baseline using one-versus-rest binary cross entropy with an additional eighth class for `No Defect`. For consistency and generalizability, I excluded `No Defect` score from mean AUROC calculation. The latter approach worked slightly better.

**FeatureGenerating.ipynb**

I create many features and observe feature importances for XGB and RF models.

**engineering.py**

This is a module that holds our feature generating function. To make optimization easier, there's an optional input for columns to drop.

**RFOptimizing.ipynb**

I remove detrimental features from a Random Forest model and use optuna for hyperparameter selection.

**XGBOptimizing.ipynb**

I remove detrimental features from an XGBoost model and use optuna for hyperparameter selection.

**EnsembleOptimizing.ipynb**

With 5-fold CV, I predict logits from RF and XGB models. I use a dense linear range of weights to find the optimal RF-XGB ratio for logit averaging.

**Submission.ipynb**

I use the optimal logit weights to create a final submission with the testing dataset.
