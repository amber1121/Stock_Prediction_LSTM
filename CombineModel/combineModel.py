from hanModel import HANModel
from lstmModel import LSTMModel
from kerasPlot import KerasPlot
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Activation
import numpy as np
from keras.optimizers import Adam
from keras.utils import np_utils
from lstmAndHanModel import LSTMAndHanModel
from fakeDataRepository import FakeDataRepository
from realDataRepository import RealDataRepository
from company_path_generator import Filter

def createLSTMModel():
    TIME_STEP = 11
    INPUT_DIM = 5
    return LSTMModel.create(TIME_STEP, INPUT_DIM)

def createHANModel():
    return HANModel.create()

def draw(model):
    KerasPlot.draw(model, 'combine_model.png')

def createCombineModel(epochs, batch_size):
    return LSTMAndHanModel(createLSTMModel(), 
                                createHANModel(), 
                                epochs, 
                                batch_size) 

# factory method
def getDataRepository():
    return RealDataRepository()

def train(model, path_object):
    han_x_train, lstm_x_train, lstm_y_train = getDataRepository().get_train_data(path_object)
    model.train(han_x_train[:428], 
                    lstm_x_train, 
                    lstm_y_train)

def evaluate(model, path_object):
    han_x_test, lstm_x_test, lstm_y_test = getDataRepository().get_test_data(path_object)
    loss, acc = model.keras_model.evaluate([han_x_test[:98], lstm_x_test], lstm_y_test)
    return acc

if __name__ == "__main__":
    han_x_train_dir = './data/han_x_train/'
    lstm_x_train_dir = './data/lstm_x_train/'
    lstm_y_train_dir = './data/lstm_y_train/'
    han_x_test_dir = './data/han_x_test/'
    lstm_x_test_dir = './data/lstm_x_test/'
    lstm_y_test_dir = './data/lstm_y_test/'
    filter_object = Filter(han_x_train_dir, lstm_x_train_dir, lstm_y_train_dir, han_x_test_dir,
    lstm_x_test_dir, lstm_y_test_dir)
    path_objects = filter_object.get_path_objects()
    sum_of_accuracy = []
    count = 0
    batch_size = 32
    epochs = 7
    for path_object in path_objects:
        count += 1
        model = createCombineModel(epochs, batch_size)
        train(model, path_object)
        acc = evaluate(model, path_object)
        print('Accuracy: ',acc)
        sum_of_accuracy.append(acc)

   
    average_accuracy = sum(sum_of_accuracy) / count
    print('Average Accuracy: ',average_accuracy) 
    print(count)   