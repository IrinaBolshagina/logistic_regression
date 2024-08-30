'''
usage: python3 logreg_train.py <path_to_dataset> <batch_size>
example: python3 logreg_train.py datasets/dataset_train.csv datasets/thetas.csv 1
'''
from logistic_regression import LogisticRegression
from logistic_batch_stoch import LogisticRegression_bach
from stochastic_logreg import StochasticLogReg
from prepare_dataset import prepare_dataset, set_y
import pandas as pd
import numpy as np
import sys


# Write result of training - theta to a csv file
def create_theta_file(houses, file):
    df = pd.DataFrame(columns=houses)
    df.to_csv(file, index=False)

def write_thetas(thetas, house, file):
    df = pd.read_csv(file)
    df[house] = thetas
    df.to_csv(file, index=False)


    
if __name__ == "__main__":

    # Check the number of arguments
    if len(sys.argv) < 2 or len(sys.argv) > 4 or (len(sys.argv) == 3 and not sys.argv[2].isdigit()):
        print("Usage: python3 logreg_train.py <dataset> <batch_size>")
        sys.exit(1)

    # Read the dataset
    dataset = sys.argv[1]
    df = pd.read_csv(dataset)
    df = prepare_dataset(df)
    houses = df["House"].unique()

    # Split the dataset into X and y
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # Define the model for usual, stochastic or mini-batch gradient descent
    
    batch_size = int(sys.argv[2]) if len(sys.argv) == 3 else len(y)

    print("batch_size: ", batch_size)

    if batch_size == len(y):
        print('Gradient descent on the whole dataset')
        model = LogisticRegression()
        
    elif batch_size == 1:
        print('Stochastic gradient descent')
        model = StochasticLogReg()
    
    elif batch_size < len(y):
        print('Mini-batch gradient descent')
        model = LogisticRegression_bach(0.01, 1000, batch_size=batch_size)
    
    else:
        print('Enter a batch size less than the number of samples in the dataset')
        sys.exit(1)

    # Create a csv file to store thetas
    file = "datasets/thetas.csv"
    create_theta_file(houses, file)

    # Train the model for each house and write thetas to the file
    for house in houses:
        print('\n', house, '\n')
        y_house = set_y(y, house)
        thetas = model.train(X, y_house)
        write_thetas(thetas, house, file)
