# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # "0,1"

import tensorflow as tf
# # config = tf.ConfigProto(allow_soft_placement=True)
# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

# from keras.models import Model
# from keras.layers.recurrent import SimpleRNN, LSTM, GRU
# from keras.layers import CuDNNGRU
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model

# from model import create_model
# from TS_MobileNet import *
from TS_Cov_MobileNet import *

import numpy as np
import time

if __name__ == '__main__':

    # Multi_GPU = False
    Multi_GPU = 3 # GPU check
    rate = [0.005,0.003,0.001]
    # rate = [0.007]
    for l_rate in rate:
        File = "history/TCN/TrainLossOnline.txt"
        with open(File, 'a') as f:
            f.write(str(l_rate) + ":...........\n")
        File = "history/TCN/TrainLoss.txt"
        with open(File, 'a') as f:
            f.write(str(l_rate) + ":...........\n")

        if not Multi_GPU:
            ################### Single GPU ###############
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # Create New Model
            # model = TS_MobileNet(input_shape=(128, 64, 64, 1),
            #                      alpha=1.0,
            #                      depth_multiplier=1,
            #                      dropout=0.1,
            #                      pooling='avg',
            #                      classes=3)
            model = TS_ConvMobileNet(input_shape=(128, 64, 64, 1),
                                     alpha=1.0,
                                     depth_multiplier=1,
                                     dropout=0.1,
                                     pooling='avg',
                                     classes=3)
            # model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
            print(model.summary())  # 0.9
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
            batch = 10

        else:
            #################### Multi GPU #################
            import os
            print("Using multiGPU version")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # GPU check
            with tf.device("/cpu:0"):
                # initialize the model
                # m = TS_MobileNet(input_shape=(128, 64, 64, 1),
                #                  alpha=1.0,
                #                  depth_multiplier=1,
                #                  dropout=0.1,
                #                  pooling='avg',
                #                  classes=3)
                m = TS_ConvMobileNet(input_shape=(128, 64, 64, 1),
                                     alpha=1.0,
                                     depth_multiplier=1,
                                     dropout=0.1,
                                     pooling='avg',
                                     classes=3)
                m.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.0), metrics=['accuracy'])
            print(m.summary())
            model = multi_gpu_model(m, 3)  # Split batch to sub-batches # GPU check  #0.0005 0.9con   1e-6 option 0.005
            print("Optimizer: SGD")
            print(l_rate)
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=l_rate, momentum=0.2, decay=0.0),
                          metrics=['accuracy'])
            # print("Optimizer: Adam")
            # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            batch = 10 * Multi_GPU  # lr: 0.001con 0.0005uncon
            # normal : lr 0.01,momentum=0.9


        xFile = "data/train"
        yFile = "data/outputDAT/trainCls"
        xDevFile = "data/dev"
        yDevFile = "data/outputDAT/devCls"

        epoch_num = 99  # 100
        # if l_rate==0.01:
        #     epoch_num = 200
        # print(epoch_num)
        best_dev = 10
        # print(batch)
        # t1 = time.clock()
        # print(t1)
        for epoch in range(epoch_num):
            print("############### Epoch", str(epoch + 1), " #################")

            train_loss_online = []
            train_loss = []
            dev_loss = []

            print("______________Training_________________")
            for i in range(6):  ###needmod 20
                print("=============================================")
                x_train_file = xFile + str(i + 1) + ".dat"
                # print("Loading x training data from",x_train_file)
                # x_train = np.memmap(x_train_file, dtype='float', mode='r', shape=(350, 128, 64, 64, 1))
                x_train_load = np.memmap(x_train_file, dtype='float', mode='r', shape=(350, 128, 64, 64, 1))
                x_train = x_train_load.copy()
                del x_train_load
                print(x_train.shape)

                y_train_file = yFile + str(i + 1) + ".npy"
                # print("Loading y training data from", y_train_file)
                y_train_load = np.load(y_train_file)
                y_train = y_train_load.copy()
                del y_train_load
                # print(y_train.shape)

                # # TODO: Delete
                # x_train = x_train[:100] ###
                # y_train = y_train[:100] ###
                print("Training model on training data", str(i + 1) + "/6 ...")
                history = model.fit(x_train, y_train, epochs=1, batch_size=batch, verbose=2)  # Maximum batch size:
                train_loss_online.append(history.history['loss'])
                print("Evaluating model on training data", str(i + 1) + "/6 ...")
                eval = model.evaluate(x_train, y_train, batch_size=batch, verbose=2)
                train_loss.append(eval[0])
                del x_train
                del y_train

            print(train_loss_online)
            train_loss_online = np.average(train_loss_online)
            print(train_loss_online)
            File = "history/TCN/TrainLossOnline.txt"
            with open(File, 'a') as f:
                f.write(str(train_loss_online) + "\n")

            # print(train_loss)
            train_loss = np.average(train_loss)
            print(train_loss)
            # Write loss result to file
            File = "history/TCN/TrainLoss.txt"
            with open(File, 'a') as f:
                f.write(str(train_loss) + "\n")


