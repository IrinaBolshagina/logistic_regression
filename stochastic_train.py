'''
usage: python3 stochastic_train.py <path_to_dataset> <path_to_theta_file>
example: python3 stochastic_train.py datasets/dataset_train.csv datasets/thetas_stochastic.csv
'''

import sys
sys.path.append('../prepare_dataset.py')
from prepare_dataset import prepare_dataset, set_y
from stochastic_logreg import StochasticLogReg
from logistic_batch_stoch import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



# Write result of training - theta to a csv file
def create_theta_file(houses, file):
    df = pd.DataFrame(columns=houses)
    df.to_csv(file, index=False)

def write_thetas(thetas, house, file):
    df = pd.read_csv(file)
    df[house] = thetas
    df.to_csv(file, index=False)


if __name__ == "__main__":

    dataset = sys.argv[1]
    df = pd.read_csv(dataset)
    df = prepare_dataset(df)
    houses = df["House"].unique()
    # houses = ['Slytherin', 'Ravenclaw', 'Gryffindor', 'Hufflepuff']

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # model = StochasticLogReg()
    model = LogisticRegression()


    file = "datasets/thetas_stochastic.csv"
    create_theta_file(houses, file)
    for house in houses:
        print('\n', house, '\n')
        y_house = set_y(y, house)
        # thetas = model.stochastic_train(X, y_house)
        thetas = model.train(X, y_house)
        write_thetas(thetas, house, file)
    
    print(thetas)

