from logistic_regression import LogisticRegression
from prepare_dataset import prepare_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys


if __name__ == "__main__":
    
    # Read test dataset, prepare it
    dataset = sys.argv[1]
    df = pd.read_csv(dataset)
    df = prepare_dataset(df)
    X = df.iloc[:, 1:]
    # print(X.describe())
    # y = df.iloc[:, 0]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Create model
    model = LogisticRegression()

    # Get result of the training - thetas - from a file 
    file_theta = sys.argv[2]
    theta = pd.read_csv(file_theta)
    houses = theta.columns.to_list()
    print(houses)

    # Predict the house using model 4 times: one-vs-all
    # for house in houses:
    #     print('\n', house, '\n')
    #     y_house = set_y(y, house)
    #     y_pred = model.predict_prob(X, thetas[house])
    #     y_class = [1 if i > 0.5 else 0 for i in y_pred]
    #     accuracy = sum(y_class == y_house) / len(y_house)
    #     print(accuracy)

    # Predict the house using model 4 times: one-vs-all
    y_pred = pd.DataFrame(columns = houses)

    for house in houses:
        print('\n', house, '\n')
        print(theta[house].to_list())
        y_pred[house] = model.predict_prob(X, theta[house].to_list())


    pred = y_pred.idxmax(axis=1)
    pd.set_option('display.max_rows', None)
    print(y_pred)
    #create new columnt with sum of 1s
    y_pred['sum'] = y_pred.sum(axis=1)
    #print(y_pred)
    print(pred)