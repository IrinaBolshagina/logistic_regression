import sys
import pandas as pd
from describe import describe, ft_count, ft_min, ft_max

'''
To prepare the dataset, we need to:
1) Load the dataset
2) Extract features
3) Check for duplicates
3) Define calsses - numeric valie for each Hofwards house
4) Delete features that are not useful
5) Delete rows with nan values
6) Normalize the data to bring all the values to the same scale (0-1)
'''


# def set_classes(df):
#     houses = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
#     for i in range(len(df)):
#         df.loc[i, "House"] = houses[df.loc[i, "House"]]
#     return df

def set_classes(df, class_1):
    classes = {class_1: 1, "Other": 0}
    for i in range(len(df)):
        df.loc[i, "House"] = classes[df.loc[i, "House"]]
    return df

def remove_nan(df):
    columns = df.shape[1]

    for index, row in df.iterrows():
        if ft_count(row.tolist()) < columns:
            df = df.drop(index, axis=0)
    return df

def remove_duplicates(dataset):
    if len(dataset) != len(dataset.drop_duplicates()):
        dataset = dataset.drop_duplicates()

def normalize(lst, x):
    return (x - ft_min(lst)) / (ft_max(lst) - ft_min(lst)) 

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

    # Read the dataset
    # dataset = sys.argv[1]
    # df = pd.read_csv(dataset)

    # Extract numerical features
    df = df.drop(columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"])
    labels = ["House"] + ["Feature " + str(i+1) for i in range(len(df.columns) - 1)]
    df.columns = labels
    

    # Check for duplicates
    df = df.drop_duplicates()

    # Delete features that are not useful
    df = df.drop(columns = ["Feature 4", "Feature 11"])

    
  
    # Set numeric values for classes
    # df = set_classes(df)
    
    # remove nan values
    df = df.dropna()
    
    # Normalize only the features
    df =  pd.concat([df["House"], normalize_dataset(df.drop(columns=["House"]))], axis=1)

    return df

    # Save the cleaned dataset to a new file
    # file_name = dataset.split(".")[0] + "_clean.csv"
    # df.to_csv(file_name, index=False)
    
    # print("Balance of classes in the dataset:", df["House"].value_counts(normalize=True), "\n")
    # print(df)
    # print("\nDataset is ready in file:", file_name)