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

def createLSTMModel():
    TIME_STEP = 11
    INPUT_DIM = 5
    return LSTMModel.create(TIME_STEP, INPUT_DIM)

def createHANModel():
    return HANModel.create()

def draw(model):
    KerasPlot.draw(model, 'combine_model.png')

# factory method
def getDataRepository():
    return RealDataRepository()

if __name__ == "__main__":
    batch_size = 8
    epochs = 20
    model = LSTMAndHanModel(createLSTMModel(), 
                            createHANModel(), 
                            epochs, 
                            batch_size)
    # two input data, one output data
    size = 1000
    han_train_x, lstm_train_x, train_y = getDataRepository().getData()
    model.train(han_train_x[:170], 
                lstm_train_x[:170], 
                train_y[:170])
    draw(model.keras_model)
    loss, acc = model.keras_model.evaluate([han_train_x[170:241], lstm_train_x[170:241]], train_y[170:241])
    print('Loss: ',loss)
    print('Accuracy: ',acc)
    