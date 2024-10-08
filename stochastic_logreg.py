import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class StochasticLogReg:

    def __init__(self, learning_rate=0.01, epochs=100, theta = None):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # add a column of ones to the input matrix
    def add_ones(self, X):
        return(np.insert(X, 0, 1, axis=1))

    # predict function
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))
    
    # loss function with logarithms
    def stochastic_logloss(self, y, y_pred):
        epsilon = 1e-15     # to avoid log(0)
        logloss1 =  y * np.log(y_pred + epsilon)
        logloss0 = (1 - y) * np.log(1 - y_pred + epsilon)
        return -1 * (logloss1 + logloss0)

    # stochastic gradient descent for one sample
    def stochastic_gradient(self, x, yi, theta):
        y_pred = self.sigmoide(np.dot(x, theta))
        grad = (y_pred - yi) * x
        return grad
    
    def train(self, X, y):
        X = X.to_numpy()
        y = y.to_numpy()
        X = self.add_ones(X)
        theta = np.zeros(X.shape[1])

        pic_cost = []
        pic_epoch = []#list[range(self.epochs)]

        for epoch in range(self.epochs):
            for i in range(len(y)):
                j = np.random.randint(len(y))
                x = X[j]
                yi = y[j]
                gradient = self.stochastic_gradient(x, yi, theta)
                theta -= gradient * self.learning_rate
                if np.linalg.norm(theta) < 1e-5:
                    break
            
            #dasha_______________
            y_pr = self.sigmoide(np.dot(x, theta))
            aaa = self.stochastic_logloss(yi, y_pr)
            pic_cost.append(aaa)
            pic_epoch.append(epoch)
            #dasha________________

            if epoch % 10 == 0:
                print(f'epoch: {epoch}')
                y_pred = self.sigmoide(np.dot(x, theta))
                print(1 if y_pred > 0.5 else 0)
                print('loss:', self.stochastic_logloss(yi, y_pred))
                print('theta:', theta)
                print()

        #dasha_______________
        fig, ax = plt.subplots()
        ax.plot(pic_epoch, pic_cost)
        plt.xlabel("epochs")
        plt.ylabel("logloss")
        plt.show()
        #dasha____________
        
        return theta


    # predict the probability of the input being in class 1
    def predict_prob(self, X, theta):
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        return self.sigmoide(np.dot(X, theta))

    def predict_class(self, X, theta):
        y_pred = self.predict_prob(X)
        return [1 if i > 0.5 else 0 for i in y_pred]

      

