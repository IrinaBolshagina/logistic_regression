import pandas as pd
import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=5000, theta = None):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # add a column of ones to the input matrix
    def add_ones(self, X):
        return(np.insert(X, 0, 1, axis=1))

    # predict function
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))
    
    # loss function with logarithms
    def logloss(self, y, y_pred):
        epsilon = 1e-15     # to avoid log(0)
        logloss1 =  y * np.log(y_pred + epsilon)
        logloss0 = (1 - y) * np.log(1 - y_pred + epsilon)
        return -1/len(y) * sum(logloss1 + logloss0)

    # gradient descent
    def gradient(self, X, y, theta):
        y_pred = self.sigmoide(np.dot(X, theta))
        grad = np.dot(X.T, (y_pred - y)) / len(y)
        return grad
    
    # train the model
    def train(self, X, y):
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        theta = np.zeros(X.shape[1])
        for epoch in range(self.epochs):
            theta_tmp = self.learning_rate * self.gradient(X, y, theta)
            # shift to the negative side of the gradient
            theta = theta - theta_tmp
            # stop if the change is too small
            if np.linalg.norm(theta) < 1e-5:
                break
            # visualization of the loss function
            if epoch % 1000 == 0:
                print(f'epoch: {epoch}')
                y_pred = self.sigmoide(np.dot(X, theta))
                print(f'loss: {self.logloss(y, y_pred)}')
                y_class = [1 if i > 0.5 else 0 for i in y_pred]
                # print('y_class:', y_class)
                accuracy = sum(y_class == y) / len(y)
                # print('sum:', sum(y_class == y), 'len:', len(y))
                print(f'accuracy: {accuracy}')
                print('theta:', theta)
                print()
        return theta

    # predict the probability of the input being in class 1
    def predict_prob(self, X, theta):
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        return self.sigmoide(np.dot(X, theta))

    def predict_class(self, X, theta):
        y_pred = self.predict_prob(X)
        return [1 if i > 0.5 else 0 for i in y_pred]

      

