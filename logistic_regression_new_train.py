from logistic_regression_new import LogisticRegression
from sklearn.model_selection import train_test_split
from prepare_dataset import prepare_dataset, set_y
import pandas as pd
import sys


# Write result of training - theta to a csv file
def create_theta_file(houses, file):
    df = pd.DataFrame(columns=houses)
    df.to_csv(file, index=False)


def write_thetas(thetas, house, file):
    df = pd.read_csv(file)
    df[house] = thetas

    df.to_csv(file, index=False)

def check_bonus(wich_grad):
    bonus = -1
    if wich_grad != "bonus":
        print("sorry, you can use just 'bonus' arguments")
        exit()
    if wich_grad == "bonus":
        print('len of your dataset:', df.shape[0])
        size = input("Please write a positive size of your batch(integer): but less than len of you dataset:")
        while isinstance(int(size), int) == False:
            size = input("Please write a positive size of your batch(integer) but less than len of you dataset:", )
        size_of_batch = int(size)
        if size_of_batch == 1:
            bonus = 1
            print("you are going to use a common stochastic gradient descent")
        elif (size_of_batch > 1 and size_of_batch < int(df.shape[0])):
            print(df.shape[1])
            print("you are going to use a batch gradient descent")
            bonus = 1
        else:
            print("your size is not right!")
            exit()
    return bonus, size_of_batch, size

if __name__ == "__main__":
    arguments = len(sys.argv)
    print("arguments=" , arguments)
    bonus = -100
    if arguments < 1:
        exit("something went wrong")
    if arguments == 1:
        exit("you didn't write the dataset")
    if arguments > 3:
        print("Sorry, I don't understand what you want...")
        exit()
    else:
        try:
            dataset = sys.argv[1]
            df = pd.read_csv(dataset)
            df = prepare_dataset(df)
        except:
            print("An exception occurred, check your dataset")
            exit()
        else:
            if arguments == 2:
                bonus = 0
            if arguments == 3:
                wich_grad = sys.argv[2]
                bonus, size_of_batch, size = check_bonus(wich_grad)
                if bonus < 0:
                    print("something went wrong")
                    exit()
    houses = df["House"].unique()
    # houses = ['Slytherin', 'Ravenclaw', 'Gryffindor', 'Hufflepuff']

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    if bonus == 1:
        file = "thetas_mini_batch_" + size + ".csv"
    elif bonus == 0:
        print("you are going to use a common gradient descent")
        file = "thetas.csv"
    else:
        exit("something went wrong, please try again")
    print(file)
    create_theta_file(houses, file)
    for house in houses:
        print('\n', house, '\n')
        y_house = set_y(y_train, house)
        if bonus:
            thetas = model.batch_train(X_train, y_house, size_of_batch)
        elif bonus == 0:
            thetas = model.train(X_train, y_house)
        else:
            exit("something went wrong, please try again")
        write_thetas(thetas, house, file)

