'''
This file is used to prepare the dataset for the machine learning model

1) Extract numerical features
2) Delete not numeric features
3) Check for duplicates
4) Delete features that are not useful
5) Delete rows with nan values
6) Normalize the data to bring all the values to the same scale (0-1)
'''

import pandas as pd
from describe import ft_min, ft_max, ft_mean

# Normalize one value to the range 0-1
def normalize(lst, x):
    return (x - ft_min(lst)) / (ft_max(lst) - ft_min(lst)) 

# Normalize all the values in the dataset
def normalize_dataset(df):
    for column in df.columns:
        col = df[column].tolist()
        col_norm = [normalize(col, x) for x in col]
        df.loc[:, column] = col_norm
    return df

# Replace nan values with the mean of the column
def replace_nan(df):
    for column in df.columns:
        col = df[column].tolist()
        new_col = [ft_mean(col) if x!= x else x for x in col]
        df.loc[:, column] = new_col
    return df

# set the y values to 1 for the house we are predicting and 0 for the rest 3 houses
def set_y(y, house):
    y = y.apply(lambda x: 1 if x == house else 0)
    return y


def prepare_dataset(df):

    # Extract numerical features
    df = df.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    labels = ["House"] + ["Feature " + str(i+1) for i in range(len(df.columns) - 1)]
    df.columns = labels

    # Delete features that are not useful
    df = df.drop(columns = ["Feature 1", "Feature 4", "Feature 11"])

    # replace nan values with the mean of the column
    df = pd.concat([df["House"], replace_nan(df.drop(columns=["House"]))], axis=1)
    
    # Normalize only the features
    df = pd.concat([df["House"], normalize_dataset(df.drop(columns=["House"]))], axis=1)

    print(df.shape)

    return df
