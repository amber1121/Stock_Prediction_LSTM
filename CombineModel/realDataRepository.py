import numpy as np
from keras.utils import np_utils

class RealDataRepository:
    def __init__(self):
        pass

    def get_train_data(self, pathObject):
        han_x_train = np.load(pathObject.get_han_x_train())
        lstm_x_train_data = np.load(pathObject.get_lstm_x_train())
        lstm_y_train_data = np.load(pathObject.get_lstm_y_train())
        lstm_y_train_data = self.one_hot_encoding(lstm_y_train_data)
        return han_x_train, lstm_x_train_data, lstm_y_train_data

    def get_test_data(self, pathObject):
        han_x_test = np.load(pathObject.get_han_x_test())
        lstm_x_test_data = np.load(pathObject.get_lstm_x_test())
        lstm_y_test_data = np.load(pathObject.get_lstm_y_test())
        lstm_y_test_data = self.one_hot_encoding(lstm_y_test_data)
        return han_x_test, lstm_x_test_data, lstm_y_test_data

    def one_hot_encoding(self, train_y):
        return np_utils.to_categorical(train_y, num_classes = 3) 

if __name__ == "__main__":
    repo = RealDataRepository()
    han, lstm, han_y = repo.getData()
    print(han.shape)
    print(lstm.shape)
    print(han_y.shape)
    