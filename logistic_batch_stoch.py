import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression_bach:

    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

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


    # def train(self, X, y):
    #     # Преобразование входных данных
    #     X = X.to_numpy()
    #     X = self.add_ones(X)
    #     y = y.to_numpy()
    #     theta = np.zeros(X.shape[1])

    #     # Списки для хранения значений потерь и эпох
    #     pic_cost = []
    #     pic_epoch = []

    #     for epoch in range(self.epochs):
    #         # Перемешивание данных в начале каждой эпохи
    #         indices = np.random.permutation(len(y))
    #         X_shuffled = X[indices]
    #         y_shuffled = y[indices]

    #         for i in range(len(y)):
    #             # Выбираем одну строку данных
    #             x_i = X_shuffled[i:i+1]
    #             y_i = y_shuffled[i:i+1]

    #             # Вычисляем градиент и обновляем веса
    #             gradient = self.gradient(x_i, y_i, theta)
    #             theta -= self.learning_rate * gradient

    #         # Вычисление потерь и точности в каждой эпохе
    #         y_pred = self.sigmoide(np.dot(X, theta))
    #         logloss_value = self.logloss(y, y_pred)
    #         pic_cost.append(logloss_value)
    #         pic_epoch.append(epoch)

    #         # Вывод значений каждые 10 эпох
    #         if epoch % 10 == 0:
    #             print(f'Epoch: {epoch}')
    #             print(f'Loss: {logloss_value}')
    #             y_class = [1 if i > 0.5 else 0 for i in y_pred]
    #             accuracy = np.mean(y_class == y)
    #             print(f'Accuracy: {accuracy}')
    #             print(f'Theta: {theta}')
    #             print()

    #     # Построение графика функции потерь
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(pic_epoch, pic_cost, label='Log-loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Log-loss')
    #     plt.title('Log-loss over epochs')
    #     plt.legend()
    #     plt.show()

    #     return theta

    def train(self, X, y):

        X = X.to_numpy()
        X = self.add_ones(X)
        y = y.to_numpy()
        theta = np.zeros(X.shape[1])    
        
        # For plots
        pic_cost = []
        pic_epoch = []

        for epoch in range(self.epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.random.permutation(len(y))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, len(y), self.batch_size):
                # Create a batch from the shuffled data
                end_i = i + self.batch_size
                X_batch = X_shuffled[i:end_i]
                y_batch = y_shuffled[i:end_i]

                gradient = self.gradient(X_batch, y_batch, theta)
                theta -= self.learning_rate * gradient

            # Calculate loss and accuracy in each epoch
            y_pred = self.sigmoide(np.dot(X, theta))
            logloss_value = self.logloss(y, y_pred)
            pic_cost.append(logloss_value)
            pic_epoch.append(epoch)

            # Print values every 80 epochs
            if epoch % 80 == 0:
                print(f'Epoch: {epoch}')
                print(f'Loss: {logloss_value}')
                y_class = [1 if i > 0.5 else 0 for i in y_pred]
                accuracy = np.mean(y_class == y)
                print(f'Accuracy: {accuracy}')
                print(f'Theta: {theta}')
                print()

        # Plot the loss function
        plt.figure(figsize=(10, 6))
        plt.plot(pic_epoch, pic_cost, label='Log-loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log-loss')
        plt.title('Log-loss over epochs')
        plt.legend()
        plt.show()

        return theta
        

    # predict the probability of the input being in class 1
    def predict_prob(self, X, theta):
        X = self.add_ones(X)  # Add this line to update X with the column of ones
        return self.sigmoide(np.dot(X, theta))

    def predict_class(self, X, theta):
        y_pred = self.predict_prob(X)
        return [1 if i > 0.5 else 0 for i in y_pred]

      

