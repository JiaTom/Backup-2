import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # "0,1"

import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.9

# from keras.models import Model
# from keras.layers.recurrent import SimpleRNN, LSTM, GRU
# from keras.layers import CuDNNGRU
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model

from model import create_model

import numpy as np
import time

if __name__ == '__main__':

    # Load Model
    modelFile = "model/Model84-Jun27_16:27:18.h5"
    model = load_model(modelFile)
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.8), metrics=['accuracy'])
    print(model.summary())

    xTestFile = "data/test"
    yTestFile = "data/outputDAT/testCls"

    eval=[]

    for i in range(2):
        x_test_file = xTestFile + str(i + 1) + ".dat"
        # print("Loading x training data from",x_train_file)
        sample_num = 350
        if i == 1:
            sample_num = 150

        x_test_load = np.memmap(x_test_file, dtype='float', mode='r', shape=(sample_num, 128, 64, 64, 1))
        x_test = x_test_load.copy()
        del x_test_load
        print(x_test.shape)

        y_test_file = yTestFile + str(i + 1) + ".npy"
        # print("Loading y training data from", y_train_file)
        y_test_load = np.load(y_test_file)
        y_test = y_test_load.copy()
        del y_test_load
        print(y_test.shape)

        pred = model.predict(x_test,batch_size=3,verbose=2)
        print(pred.shape)
        pred = np.argmax(pred,axis=1)
        y_test = np.argmax(y_test,axis=1)

        print(pred.shape,y_test.shape)
        eval.append(np.sum(pred==y_test)/len(pred))

    print(eval)
