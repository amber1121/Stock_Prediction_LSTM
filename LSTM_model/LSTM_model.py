import pandas as pd
import numpy as np
import tensorflow as tf
from preprocessing_normalize import DataProcessing
import pandas_datareader.data as pdr
import yfinance as fix
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.optimizers import RMSprop
import matplotlib
import matplotlib.pyplot as plt
from keras import backend as K 
from keras.layers import Input, Dense, concatenate, LSTM, Dropout
from keras.models import Model
from keras.optimizers import adam
import os

def create_model(TIME_STEP, INPUT_DIM):
    inputs = Input(shape=(TIME_STEP,INPUT_DIM))
    x = LSTM(200, return_sequences = True)(inputs)
    x = Dropout(0.4)(x)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)
    x = Dense(3, activation="softmax")(x)
    model = Model(inputs, x)
    opt = adam(lr=0.0005, decay=1e-6)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model

def read_file(fileName, train_ratio):
    process = DataProcessing(fileName, train_ratio)
    return process

def split_datasets(process, TIME_STEP, INPUT_DIM):
    SEQUENCE_LENGTH = 10
    return process.split_data(SEQUENCE_LENGTH)

def save(file_name, np_file):
    np.save(file_name, np_file)

def plot_loss(model, companyName):
    hh = model.fit(x_train, y_train, epochs=30, validation_split=0.1, batch_size=32)
    plt.figure()
    plt.plot(hh.history['loss'])
    plt.plot(hh.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='upper left')
    plt.savefig(companyName)
    plt.show()

if __name__ == "__main__":
    INPUT_DIM = 5
    TIME_STEP = 11
    # current path:C:\Users\Lab1424\Desktop\Stock_Prediction_LSTM\LSTM_model 
    os.chdir('C:\\Users\\Lab1424\\Desktop\\Stock_Prediction_LSTM')
    os.getcwd()
    file_path = "./format_csv_file_sp500/"
    count = 0
    sum_accuracy = []
    for company in os.listdir(file_path):
        process = read_file(file_path + company, 0.8)
        x_train,y_train,x_test,y_test = split_datasets(process,TIME_STEP, INPUT_DIM)
        if len(x_train) != 428 or len(y_train)!= 428 or len(x_test) != 98 or len(y_test)!=98:
            continue
        # company_name = company[:-4] 
        # np.save('./LSTM_model/x_train/'+ company_name + '_x_train', x_train)
        # np.save('./LSTM_model/y_train/' + company_name + '_y_train', y_train)
        # np.save('./LSTM_model/x_test/' + company_name + '_x_test', x_test)
        # np.save('./LSTM_model/y_test/' + company_name + '_y_test', y_test)
        model = create_model(TIME_STEP, INPUT_DIM)
        y_train = np_utils.to_categorical(y_train, num_classes=3)
        y_test = np_utils.to_categorical(y_test, num_classes = 3) 
        # op=[EarlyStopping(monitor='val_loss',min_delta=0.0001,mode='min',verbose=2,patience=200)]   
        history = model.fit(x_train, y_train,epochs=10)
        loss, accuracy = model.evaluate(x_test, y_test)
        count += 1
        sum_accuracy.append(accuracy)
        # print("test loss: {}".format(loss))
        print('test accuracy: {}'.format(accuracy))
    
    total_accuracy = sum(sum_accuracy)
    print('total size: ',count)
    print('Average accuray: ', total_accuracy/count)
    