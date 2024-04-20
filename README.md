# Kaggle S4:E3 Competition (Steel Plate Defect Detection)

## Introduction

This repository documents my submission to Kaggle's [Season 4, Episode 3 Playground Series Competition](https://www.kaggle.com/competitions/playground-series-s4e3/leaderboard). My final submission was an ensemble of the submission given here and [this](https://www.kaggle.com/code/arunklenin/ps4e3-steel-plate-fault-prediction-multilabel) public notebook. I finished 135 out of 2199 teams, in the top 7%.

Kaggle's Playground Series gives people the opportunity to build competitive models with tabular data. This competition used synthetic data generated from [UCI's Steel Plate Defect Dataset](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) and asked competitors to predict the defects of steel plates across 7 categories. We were given a training dataset and a test dataset without labels. A public score is displayed when submissions are made, but this is calculated on a fraction of the submission. The true score is hidden until the end of the competition to prevent probing and incentivize solid validation schemes.

## Method & Results

To get an understanding of the data, I start with an exploratory data analysis. Since there are many features, I separate their analyses into geometric and non-geometric variables. 

I then establish an XGBoost baseline of `0.88444` and improve it to `0.88454` after creating an eighth class for `No Defect`. For consistency and generalizability to the competition, I excluded the `No Defect` score from mean AUROC calculation. Based on my EDA, I generated many new features and created PCA features with these. (Note that for feature engineering on the test set, I generate PCA features from the principal components determined on the train set.)

I build two models, Random Forest and XGBoost, with the intent to ensemble them. Because they are diverse, I hypothesized that this selection would boost the performance. For the XGBoost model, I used leave-one-out feature selection to remove features until removing anymore hurt the 5-fold CV score. I did the same for my RF model, but only a few were dropped.

I used the optuna library in phases to choose optimal hyperparameters for my XGB and RF models. I was able to achieve a 5-fold CV AUROC score of `0.88648` for my XGB model and `0.87965` for my RF model. Then, for each of the five folds, I found the weighted average of logit predictions that maximized the AUROC on validation. My final, 5-fold CV AUROC score was `0.88889`.

## Note on Random Forest

Many competitors found that RF performed badly against XGBoost, CatBoost, and LGBM models. The latter models work incredibly well on simple features; this is why they show great baseline scores. However, RF works by using large averages of diverse decision trees to drive variance down. In my approach, the use of many features, including a full army of principal components, allowed the decision trees in the random forest to get the same information in new ways. The diversity created across the trees allowed the RF model to perform increasingly well with scale. Even though the RF model performed significantly worse than the XGB model, it was able to push our best XGB score of `0.88648` to `0.88889` in the final ensemble.

## Files

**EDA.ipynb**

This is a notebook where I perform basic exploratory data analysis on the dataset. I found that the targets were not mutually exclusive (a few had two defects), and a good fraction of datapoints had no defect.

**BaseScores.ipynb**

Here, I analyze the performance of XGBoost with 5-fold CV before doing any feature engineering. I also gave a baseline using one-versus-rest binary cross entropy with an additional eighth class for `No Defect`. The latter approach worked slightly better.

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
