from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras import backend as K

class HANModel():
    def __init__(self):
        pass
    @staticmethod
    def create():
        input1 = Input(shape=(40, 200), dtype='float32')
        dense_layer = Dense(200, activation='tanh')(input1)
        softmax_layer = Activation('softmax')(dense_layer)
        attention_mul = multiply([softmax_layer,input1])
        vec_sum = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
        pre_model1 = Model(input1, vec_sum)
        pre_model2 = Model(input1, vec_sum)
        input2 = Input(shape=(11, 40, 200), dtype='float32')
        pre_gru = TimeDistributed(pre_model1)(input2)
        l_gru = Bidirectional(GRU(100, return_sequences=True))(pre_gru)
        post_gru = TimeDistributed(pre_model2)(l_gru)
        dense1 = Dense(100, activation='tanh')(post_gru)
        model = Model(input2, dense1)
        return model

if __name__ == "__main__":
    model = HANModel.create()
    model.summary()