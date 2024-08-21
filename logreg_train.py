from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from prepare_dataset import prepare_dataset, set_y
import pandas as pd
import numpy as np
import sys


# Write result of training - theta to a csv file
def create_theta_file(houses, file):
    df = pd.DataFrame(columns=houses)
    df.to_csv(file, index=False)

def write_thetas(thetas, house, file):
    df = pd.read_csv("thetas.csv")
    df[house] = thetas
    df.to_csv(file, index=False)

# def train_each_house(X, y, model, file, houses):
    



if __name__ == "__main__":

    dataset = sys.argv[1]
    df = pd.read_csv(dataset)
    df = prepare_dataset(df)
    houses = df["House"].unique()
    print(houses)

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    model = LogisticRegression()
    
    
    # y = set_y(y, 'Hufflepuff')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    # thetas = model.train(X_train, y_train)
    # create_theta_file()
    # write_thetas(thetas, 'Hufflepuff')

    # y_pred = model.predict_prob(X_test)
    # print(y_pred)

    # y_class = [1 if i > 0.5 else 0 for i in y_pred]
    # print(y_class)

    # accuracy = sum(y_class == y_test) / len(y_test)
    # print(accuracy)

    file = "thetas.csv"
    create_theta_file(houses, file)
    for house in houses:
        print('\n', house, '\n')
        y_house = set_y(y, house)
        X_train, X_test, y_train, y_test = train_test_split(X, y_house, test_size=0.2, stratify=y_house)
        thetas = model.train(X_train, y_train)
        write_thetas(thetas, house, file)
