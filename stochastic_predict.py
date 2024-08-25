'''
usage: python3 stochastic_predict.py <path_to_dataset> <path_to_theta_file>
example: python3 stochastic_predict.py datasets/dataset_test.csv datasets/thetas_stochastic.csv
'''

from stochastic_logreg import StochasticLogReg
from prepare_dataset import prepare_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
import sys


if __name__ == "__main__":
    
    # Read test dataset, prepare it
    dataset = sys.argv[1]
    df = pd.read_csv(dataset)
    df = prepare_dataset(df)
    X = df.iloc[:, 1:]

    # Create model
    model = StochasticLogReg()

    # Get result of the training - thetas - from a file 
    file_theta = sys.argv[2]
    theta = pd.read_csv(file_theta)
    houses = theta.columns.to_list()

    # Predict the house using model 4 times: one-vs-all
    y_pred = pd.DataFrame(columns = houses)

    for house in houses:
        y_pred[house] = model.predict_prob(X, theta[house].to_list())

    pred = y_pred.idxmax(axis=1)

    predictions = pd.DataFrame(columns = ['Index','Hogwarts House'])
    predictions['Hogwarts House'] = pred
    predictions['Index'] = pd.Series(range(len(predictions)))
    

    # write the result to a csv file
    file_res = "datasets/predictions.csv"
    predictions.to_csv(file_res, index=False)

    print("Result in file:", file_res)



    # For testing with X_test and y_test
    X_test = pd.read_csv("datasets/X_test.csv")
    y_test = pd.read_csv("datasets/y_test.csv")
    y_test = y_test.iloc[:, 0]
    
    y_pred = pd.DataFrame(columns = houses)

    for house in houses:
        y_pred[house] = model.predict_prob(X_test, theta[house].to_list())

    pred = y_pred.idxmax(axis=1)
    accuracy = sum(pred == y_test) / len(y_test)
    print('\nAccuracy:', accuracy)

    