from hanModel import HANModel
from lstmModel import LSTMModel
from kerasPlot import KerasPlot
from keras.layers import concatenate
from keras.layers.core import Dense, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Activation
import numpy as np
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.optimizers import adam
from keras import initializers
from keras import regularizers

class LSTMAndHanModel:
    def __init__(self, lstm, han, epochs, batch_size):
        self.keras_model = self.combine(han, lstm)
        self.batch_size = batch_size
        self.epochs = epochs
    # private method
    def combine(self, han, lstm):
        combinedInput = concatenate([han.output, lstm.output])
        #x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combinedInput)
        x = Dense(3, activation='tanh')(combinedInput)
        x = Activation('softmax')(x)
        return Model(inputs=[han.input, lstm.input], outputs=x)

    def train(self, han_train_x, lstm_train_x, train_y):
        self.compile()
        self.fit(han_train_x, lstm_train_x, train_y)

    def compile(self):
        # opt = Adam(lr=1e-3, decay=1e-3 / 200)
        opt = adam(lr=0.0005, decay=1e-6)
        self.keras_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    def fit(self, han_train_x, lstm_train_x, train_y):
        self.keras_model.fit([han_train_x, lstm_train_x], 
                            train_y, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, validation_split=0.1)

    def predict(self, han_x, lstm_x):
        return self.keras_model.predict([han_x, lstm_x])