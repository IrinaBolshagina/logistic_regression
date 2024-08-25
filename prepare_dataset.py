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
from describe import ft_min, ft_max

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

# set the y values to 1 for the house we are predicting and 0 for the rest 3 houses
def set_y(y, house):
    y = y.apply(lambda x: 1 if x == house else 0)
    return y


def prepare_dataset(df):

    # Extract numerical features
    df = df.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    labels = ["House"] + ["Feature " + str(i+1) for i in range(len(df.columns) - 1)]
    df.columns = labels
    
    # Check for duplicates
    df = df.drop_duplicates()

    # Delete features that are not useful
    df = df.drop(columns = ["Feature 1", "Feature 4", "Feature 11"])

    # remove nan values from everywhere but the house column
    df = df.dropna(subset=df.columns[1:])
    
    # Normalize only the features
    df = pd.concat([df["House"], normalize_dataset(df.drop(columns=["House"]))], axis=1)

    return df
