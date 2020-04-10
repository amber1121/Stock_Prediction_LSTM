import numpy as np
from keras.utils import np_utils


class FakeDataRepository:
    def __init__(self):
        pass

    def getData(self):
        han_train_x = self.get_han_train_x(1000)
        lstm_train_x = self.get_lstm_train_x(1000)
        train_y = self.get_train_y(1000)
        return han_train_x, lstm_train_x, train_y

    def get_lstm_train_x(self, size):
        return np.random.random((size, 11, 2))

    def get_han_train_x(self, size):
        return np.random.random((size, 11, 40, 200))

    def one_hot_encoding(self, train_y):
        return np_utils.to_categorical(train_y, num_classes = 3) 

    def get_train_y(self, size):
        train_y = np.random.randint(-1,1, size)
        return self.one_hot_encoding(train_y)
