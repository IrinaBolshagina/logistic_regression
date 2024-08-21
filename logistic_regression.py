import pandas as pd
import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=5000, theta = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = theta

    # add a column of ones to the input matrix
    def add_ones(self, X):
        return(np.insert(X, 0, 1, axis=1))
    
    # calculate weights for classes 0 and 1
    def get_weights(self, y):
        weight0 = len(y[y == 0])/ len(y)
        weight1 = len(y[y == 1])/ len(y)
        return (weight0, weight1)

    # predict function
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))
    
    # loss function with weights
    def logloss(self, y, y_pred, w0 = 1, w1 = 1):
        epsilon = 1e-15     # to avoid log(0)
        logloss1 =  y * np.log(y_pred + epsilon) * w1 
        logloss0 = (1 - y) * np.log(1 - y_pred + epsilon) * w0
        return -1/len(y) * sum(logloss1 + logloss0)

    # gradient descent
    def gradient(self, X, theta, y):
        y_pred = self.sigmoide(np.dot(X, theta))  # check format that is returned
        grad = np.dot(X.T, (y_pred - y)) / len(y)
        # print(type(grad))
        return grad
    
    # train the model
    def train(self, X, y):
        self.add_ones(X)
        theta_new = np.zeros(X.shape[1])
        weight0, weight1 = self.get_weights(y)
        for _ in range(self.epochs):
            theta = theta_new
            # shift to the negative side of the gradient
            theta_new = theta - self.learning_rate * self.gradient(X, theta, y)
            # stop if the change is too small
            if np.linalg.norm(theta_new - theta) < 1e-5:
                break
            # visualization of the loss function
            if _ % 700 == 0:
                print(f'epoch: {_}')
                y_pred = self.sigmoide(np.dot(X, theta))
                print(f'loss: {self.logloss(y, y_pred, weight0, weight1)}')
                y_class = [1 if i > 0.5 else 0 for i in y_pred]
                # print('y_class:', y_class)
                accuracy = sum(y_class == y) / len(y)
                # print('sum:', sum(y_class == y), 'len:', len(y))
                print(f'accuracy: {accuracy}')
                print('theta:', theta)
                print()
        self.theta = theta_new
        return self.theta


    # predict the probability of the input being in class 1
    def predict_prob(self, X, theta):
        self.add_ones(X)
        return self.sigmoide(np.dot(X, theta))

    # Write thetas to a file
    def write_thetas(self, file_name, theta):
        with open(file_name, 'w') as f:
            f.write(','.join(map(str, theta)))

    def predict_class(self, X, theta):
        y_pred = self.predict_prob(X)
        return [1 if i > 0.5 else 0 for i in y_pred]

      

