from keras.layers import Input, Dense, concatenate, LSTM, Dropout
from keras.models import Model
from keras.optimizers import adam

class LSTMModel:
    def __init__(self):
        pass
    
    @staticmethod 
    def create(TIME_STEP, INPUT_DIM):
        inputs = Input(shape=(TIME_STEP,INPUT_DIM))
        x = LSTM(200, return_sequences = True)(inputs)
        x = Dropout(0.4)(x)
        x = LSTM(100)(x)
        x = Dropout(0.2)(x)
        return Model(inputs, x)

if __name__ == "__main__":
    model = LSTMModel.create(10, 2)
    opt = adam(lr=0.0005, decay=1e-6)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()