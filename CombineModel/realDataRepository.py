import numpy as np
from keras.utils import np_utils

class RealDataRepository:
    def __init__(self):
        pass

    def getData(self):
        han_train_data = np.load('./data/han_x_train/FB_x_train.npy')
        lstm_train_data = np.load('./data/lstm_x_train/FB_x_train_LSTMmodel.npy')
        han_y_train_data = np.load('./data/lstm_y_train/FB_y_train_LSTMmodel.npy')
        han_y_train_data = self.one_hot_encoding(han_y_train_data)
        return han_train_data, lstm_train_data, han_y_train_data

    def one_hot_encoding(self, train_y):
        return np_utils.to_categorical(train_y, num_classes = 3) 

if __name__ == "__main__":
    repo = RealDataRepository()
    han, lstm, han_y = repo.getData()
    print(han.shape)
    print(lstm.shape)
    print(han_y.shape)
    