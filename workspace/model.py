import numpy as np
import time
import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input
from keras.utils import to_categorical


# def create_model(timestep=128,height=64,width=64,channels=1):
#     inp = Input(shape=(timestep, height, width, channels))
#     x = TimeDistributed(Conv2D(128, (2, 2), padding='valid', activation='relu'))(inp)
#     # x = TimeDistributed(Conv2D(16, (4, 4), padding='valid', activation='relu'))(x)
#     x = TimeDistributed(AveragePooling2D((2, 2), padding='valid'))(x)
#     x = TimeDistributed(Conv2D(64, (2, 2), padding='valid', activation='relu'))(x)
#     x = TimeDistributed(MaxPooling2D((2, 2), padding='valid'))(x)
#     x = TimeDistributed(Flatten())(x)
#     x = CuDNNGRU(units=92, return_sequences=True)(x)
#     x = CuDNNGRU(units=50, return_sequences=False)(x)
#     x = Dropout(.2)(x)
#     x = Dense(10, activation='softmax')(x)
#     model = Model(inp, x)
#
#     return model

def create_model(timestep=128,height=64,width=64,channels=1):
    inp = Input(shape=(timestep, height, width, channels))

    x = TimeDistributed(Conv2D(16, (2, 2), padding='valid'))(inp)
    x = TimeDistributed(BatchNormalization(axis=-1, momentum=0.9997, scale=False))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(Conv2D(32, (2, 2), padding='valid'))(x)
    x = TimeDistributed(BatchNormalization(axis=-1, momentum=0.9997, scale=False))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(AveragePooling2D((4, 4), padding='valid'))(x)

    x = TimeDistributed(Conv2D(64, (2, 2), padding='same'))(x)
    x = TimeDistributed(BatchNormalization(axis=-1, momentum=0.9997, scale=False))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(Conv2D(92, (2, 2), padding='same'))(x)
    x = TimeDistributed(BatchNormalization(axis=-1, momentum=0.9997, scale=False))(x)
    x = TimeDistributed(Activation('relu'))(x)

    x = TimeDistributed(AveragePooling2D((15, 15), padding='valid'))(x)
    x = TimeDistributed(Flatten())(x)
    x = CuDNNLSTM(units=256, return_sequences=True)(x)
    x = CuDNNLSTM(units=128, return_sequences=False)(x)
    x = Dropout(.2)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inp, x)

    return model

if __name__ == '__main__':
    m = create_model()
    print(m.summary())

