from mini_batch_logreg import Mini_batch_logreg
from sklearn.model_selection import train_test_split
from prepare_dataset import prepare_dataset, set_y
import pandas as pd
import sys


# Write result of training - theta to a csv file
def create_theta_file(houses, file):
    df = pd.DataFrame(columns=houses)
    df.to_csv(file, index=False)


def write_thetas(thetas, house, file):
    df = pd.read_csv("thetas.csv")
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

    model = Mini_batch_logreg()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    file = "thetas.csv"
    create_theta_file(houses, file)
    for house in houses:
        print('\n', house, '\n')
        y_house = set_y(y_train, house)
        thetas = model.mini_batch_train(X_train, y_house, 32)
        write_thetas(thetas, house, file)

