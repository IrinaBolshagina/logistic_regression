import pandas as pd
import numpy as np


class Mini_batch_logreg:

    def __init__(self, learning_rate=0.01, epochs=5000, theta=None):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # add a column of ones to the input matrix
    def add_ones(self, X):
        return (np.insert(X, 0, 1, axis=1))

    # predict function
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    # loss function with logarithms
    def logloss(self, y, y_pred):
        epsilon = 1e-15  # to avoid log(0)
        logloss1 = y * np.log(y_pred + epsilon)
        logloss0 = (1 - y) * np.log(1 - y_pred + epsilon)
        return -1 / len(y) * sum(logloss1 + logloss0)

    # gradient descent
    def mini_batch_gradient(self, _Xt, yt, theta):
        y_pred = self.sigmoide(np.dot(_Xt, theta))
        grad = np.dot(_Xt.T, (y_pred - yt)) / len(yt)
        return grad

    # def mini_batch_gradient(self, X, y, theta):
    #     y_pred = self.sigmoide(np.dot(X, theta))
    #     grad = np.dot(X.T, (y_pred - y)) / len(y)
    #     return grad

    # train the model
    def mini_batch_train(self, X, y, len_of_batch):
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        print( "batch = ", len_of_batch)
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        datalist = []
        lendata = X.shape[1]
        # print("xshape", X.shape)
        # print("yshape", y.shape)
        i = 0
        iend = 0
        count = 0
        y= np.array(y)
        theta = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            end_batch = np.random.randint(len(y))
            if end_batch-len_of_batch <= 0:
                start_batch = 0
                end_batch = len_of_batch
            else:
                start_batch = end_batch - len_of_batch
            # print("start", start_batch)
            # print("end", end_batch)
            _Xt = X[start_batch:end_batch, :]
            print('shape x', _Xt.shape)
            yt = y[start_batch:end_batch]
            print('shape y', yt.shape)

            theta_tmp = self.learning_rate * self.mini_batch_gradient(_Xt, yt, theta)
            # shift to the negative side of the gradient
            theta = theta - theta_tmp
            # stop if the change is too small
            if np.linalg.norm(theta) < 1e-5:
                break
            # visualization of the loss function
            if epoch % 1000 == 0:
                print(f'epoch: {epoch}')
                y_pred = self.sigmoide(np.dot(_Xt, theta))
                print(f'loss: {self.logloss(yt, y_pred)}')
                y_class = [1 if i > 0.5 else 0 for i in y_pred]
                # print('y_class:', y_class)
                accuracy = sum(y_class == yt) / len(yt)
                # print('sum:', sum(y_class == y), 'len:', len(y))
                print()
                print(f'accuracy: {accuracy}')
                print('theta:', theta)
                print()
        return theta

    # predict the probability of the input being in class 1
    def predict_prob(self, X, theta):
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        return self.sigmoide(np.dot(X, theta))

    # Write thetas to a file
    def write_thetas(self, file_name, theta):
        with open(file_name, 'w') as f:
            f.write(','.join(map(str, theta)))

    def predict_class(self, X, theta):
        y_pred = self.predict_prob(X)
        return [1 if i > 0.5 else 0 for i in y_pred]



