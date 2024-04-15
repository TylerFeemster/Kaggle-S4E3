import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def create_features(df):
    new_df = df.copy()
    new_df['Height'] = np.abs(new_df['Y_Maximum'] - new_df['Y_Minimum'])
    new_df['Width'] = np.abs(new_df['X_Maximum'] - new_df['X_Minimum'])
    new_df[['Log_Outside_X_Index',
            'Log_X_Perimeter', 'Log_Y_Perimeter']] = np.log(new_df[['Outside_X_Index',
                                                                    'X_Perimeter', 'Y_Perimeter']] + 1e-6)
    new_df[['Log_Width', 'Log_Height']] = np.log(
        new_df[['Width', 'Height']] + 1)
    new_df['Abs_Orientation'] = np.abs(new_df['Orientation_Index'])
    new_df['Log_Range'] = np.log(
        1 + new_df['Maximum_of_Luminosity']) - np.log(1 + new_df['Minimum_of_Luminosity'])
    new_df['Log_Lum'] = np.log(new_df['Sum_of_Luminosity'])
    new_df['Log_Avg_Lum'] = new_df['Log_Lum'] - 2 * new_df['LogOfAreas']

    return new_df


def full_feature_engineering(train_df, test_df = None, drop_cols=['Y_Maximum', 'Y_Minimum', 'X_Maximum', 'X_Minimum']):
    
    y_cols = ['Pastry', 'Z_Scratch', 'K_Scatch',
              'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    non_cols = ['id', *y_cols]
    
    df1 = create_features(train_df)

    X = df1.drop(columns=non_cols)
    X_stats = X.describe().T[['mean', 'std']]
    X = (X - X_stats['mean']) / X_stats['std'] # normalize before pca
    pca = PCA().fit(X)

    if test_df is not None:
        test_df1 = create_features(test_df)
        X_test = test_df1.drop(columns=['id'])
        X_test = (X_test - X_stats['mean']) / X_stats['std'] # normalize with training data
        test_df1[[f'pca_{i}' for i in range(len(pca.components_))]] = pca.transform(X_test)
        return test_df1.drop(columns=drop_cols)
    
    df1[[f'pca_{i}' for i in range(len(pca.components_))]] = pca.transform(X)
    df1['No Defect'] = (df1[y_cols].sum(axis=1) == 0).astype('int')
    return df1.drop(columns=drop_cols)
