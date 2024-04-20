# Kaggle S4:E3 Competition (Steel Plate Defect Detection)

## Introduction

This repository documents my submission to Kaggle's [Season 4, Episode 3 Playground Series Competition](https://www.kaggle.com/competitions/playground-series-s4e3/leaderboard). My final submission was an ensemble of the submission given here and [this](https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel) public notebook. I finished 135 out of 2199 teams, in the top 7%.

Kaggle's Playground Series gives people the opportunity to build competitive models with tabular data. This competition used synthetic data generated from [UCI's Steel Plate Defect Dataset](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) and asked competitors to predict the defects of steel plates across 7 categories. We were given a training dataset and a test dataset without labels. A public score is displayed when submissions are made, but this is calculated on a fraction of the submission. The true score is hidden until the end of the competition. This prevents probing and incentivizes solid validation schemes.

## Method & Results



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
