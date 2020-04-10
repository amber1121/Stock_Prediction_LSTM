from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad

if __name__ == "__main__":
    # inp1 = Input(shape=(10,32))
    # inp2 = Input(shape=(10,32))
    # print('Input1:  ')
    # print(type(inp1))
    # print('input 2==>')
    # print(type(inp2))
    # cc1 = concatenate([inp1, inp2],axis=0) # Merge data must same row column
    # output = Dense(30, activation='relu')(cc1)
    # model = Model(inputs=[inp1, inp2], outputs=output)
    # model.summary()
    # merge row must same column size
    inp1 = Input(shape=(20,10))
    print('type of inp1: ', type(inp1))
    inp2 = Input(shape=(32,10))
    print('type of inp2==>', type(inp2))
    cc1 = concatenate([inp1, inp2],axis=1)
    output = Dense(30, activation='relu')(cc1)
    model = Model(inputs=[inp1, inp2], outputs=output)
    model.summary()



# first_input = Input(shape=(2, ))
# first_dense = Dense(1, )(first_input)
# print('first input: ', type(first_input))
# print('First Dense==> ', type(first_dense))
# second_input = Input(shape=(2, ))
# second_dense = Dense(1, )(second_input)

# merge_one = concatenate([first_dense, second_dense])

# third_input = Input(shape=(1, ))
# merge_two = concatenate([merge_one, third_input])

# model = Model(inputs=[first_input, second_input, third_input], outputs=merge_two)
# ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=ada_grad, loss='binary_crossentropy',
#                metrics=['accuracy'])
# model.summary()