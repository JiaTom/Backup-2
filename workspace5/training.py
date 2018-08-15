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
from TS_MobileNet import *
# from TS_Cov_MobileNet import *

import numpy as np
import time


if __name__ == '__main__':

    # Multi_GPU = False
    Multi_GPU = 3 # GPU check

    if not Multi_GPU:
        ################### Single GPU ###############
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Create New Model
        model = TS_MobileNet(input_shape=(128, 64, 64, 1),
                         alpha=1.0,
                         depth_multiplier=1,
                         dropout=0.1,
                         pooling='avg',
                         classes=3)
        # model = TS_ConvMobileNet(input_shape=(128, 64, 64, 1),
        #                          alpha=1.0,
        #                          depth_multiplier=1,
        #                          dropout=0.1,
        #                          pooling='avg',
        #                          classes=3)
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
        print(model.summary())                                                    # 0.9
        model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])
        batch = 10

    else:
        #################### Multi GPU #################
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # GPU check
        with tf.device("/cpu:0"):
            # initialize the model
            m = TS_MobileNet(input_shape=(128, 64, 64, 1),
                             alpha=1.0,
                             depth_multiplier=1,
                             dropout=0.1,
                             pooling='avg',
                             classes=3)
            # m = TS_ConvMobileNet(input_shape=(128, 64, 64, 1),
            #                      alpha=1.0,
            #                      depth_multiplier=1,
            #                      dropout=0.1,
            #                      pooling='avg',
            #                      classes=3)
            m.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01,momentum=0.0),metrics=['accuracy'])
        # m = TS_MobileNet(input_shape=(128, 64, 64, 1),
        #                  alpha=1.0,
        #                  depth_multiplier=1,
        #                  dropout=0.1,
        #                  pooling='avg',
        #                  classes=3)
        # m.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.0), metrics=['accuracy'])

        print(m.summary())
        model = multi_gpu_model(m, 3)  # Split batch to sub-batches # GPU check  # 0.001
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.2,decay=0), metrics=['accuracy'])
        batch = 10 * Multi_GPU


    # # # Load Model # current section is for TS_Mobile Net, please adjust params below if require TS_ConvMobile
    # modelFile = "SGD/model/Model0-Jul30_17:37:48.h5"
    # import keras.activations
    # keras.activations.relu6 = relu6 # write self defined.activation to keras, otherwise cannot identified
    # if Multi_GPU:
    #     batch=20
    #     import os
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    #     with tf.device("/cpu:0"): # Multi GPU
    #         m = load_model(modelFile)
    #         print(m.summary())
    #     model = multi_gpu_model(m, 2)
    #     model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.8), metrics=['accuracy'])
    # else:
    #     batch=10
    #     model = load_model(modelFile) # Single GPU
    #     print(model.summary())


    xFile = "data/train"
    yFile = "data/outputDAT/trainCls"
    xDevFile = "data/dev"
    yDevFile = "data/outputDAT/devCls"



    epoch_num = 200 #100
    best_dev = 10
    # print(batch)
    # t1 = time.clock()
    # print(t1)
    for epoch in range(epoch_num):
        print("############### Epoch",str(epoch+1)," #################")

        train_loss_online=[]
        train_loss = []
        dev_loss = []

        print("______________Training_________________")
        for i in range(6): ###needmod 20
            print("=============================================")
            x_train_file = xFile + str(i+1)+".dat"
            # print("Loading x training data from",x_train_file)
            # x_train = np.memmap(x_train_file, dtype='float', mode='r', shape=(350, 128, 64, 64, 1))
            x_train_load = np.memmap(x_train_file, dtype='float', mode='r', shape=(350, 128, 64, 64, 1))
            x_train = x_train_load.copy()
            del x_train_load
            print(x_train.shape)

            y_train_file = yFile + str(i+1)+".npy"
            # print("Loading y training data from", y_train_file)
            y_train_load = np.load(y_train_file)
            y_train = y_train_load.copy()
            del y_train_load
            # print(y_train.shape)

            # # TODO: Delete
            # x_train = x_train[:100] ###
            # y_train = y_train[:100] ###
            print("Training model on training data",str(i+1)+"/6 ...")
            history=model.fit(x_train, y_train, epochs=1, batch_size=batch, verbose=2)  # Maximum batch size:
            train_loss_online.append(history.history['loss'])
            print("Evaluating model on training data", str(i + 1) + "/6 ...")
            eval = model.evaluate(x_train,y_train,batch_size=batch, verbose=2)
            train_loss.append(eval[0])
            del x_train
            del y_train

        print(train_loss_online)
        train_loss_online = np.average(train_loss_online)
        print(train_loss_online)
        # File = "history/LRCN/TrainLossOnline.txt"
        # File = "history/TCN/TrainLossOnline.txt"
        File = "history/LRCNlr0.01momen0.9/TrainLossOnline.txt"
        with open(File, 'a') as f:
            f.write(str(train_loss_online) + "\n")

        # print(train_loss)
        train_loss = np.average(train_loss)
        print(train_loss)
        # Write loss result to file
        # File = "history/LRCN/TrainLoss.txt"
        # File = "history/TCN/TrainLoss.txt"
        File = "history/LRCNlr0.01momen0.9/TrainLoss.txt"
        with open(File, 'a') as f:
            f.write(str(train_loss) + "\n")

        print("______________Evaluating_________________")

        for i in range(2): ###needmod 5 # 3 for final
            x_dev_file = xDevFile + str(i + 1) + ".dat"
            # print("Loading x developing data from", x_dev_file)
            sample_num = 350
            if i+1 == 2:###needmod
                sample_num = 100 # 300 for final
            # x_dev = np.memmap(x_dev_file, dtype='float', mode='r', shape=(sample_num, 128, 64, 64, 1))
            x_dev_load = np.memmap(x_dev_file, dtype='float', mode='r', shape=(sample_num, 128, 64, 64, 1))
            x_dev = x_dev_load.copy()
            del x_dev_load
            print(x_dev.shape)

            y_dev_file = yDevFile + str(i + 1) + ".npy"
            # print("Loading y developing data from", y_dev_file)
            y_dev_load = np.load(y_dev_file)
            y_dev = y_dev_load.copy()
            del y_dev_load
            # print(y_dev.shape)

            # # TODO: Delete
            # x_dev = x_dev[:100]  ###
            # y_dev = y_dev[:100]  ###
            print("Evaluating model on developing data", str(i + 1) + "/2 ...")
            eval = model.evaluate(x_dev,y_dev,batch_size=batch,verbose=2)
            dev_loss.append(eval[0])
            del x_dev
            del y_dev

        # print(dev_loss)
        # print(len(dev_loss))
        # dev_loss = np.average(dev_loss)
        tmp=0
        for i in range(len(dev_loss)):
            if i+1 < 2: ###needmod
                print("regular dev:",i)
                tmp += (350/400)*dev_loss[i]
            else:
                tmp += (100/400)*dev_loss[i] ###needmod
        dev_loss = tmp
        print(dev_loss)
        # write loss result to file
        # File = "history/LRCN/DevLoss.txt"
        # File = "history/TCN/DevLoss.txt"
        File = "history/LRCNlr0.01momen0.9/DevLoss.txt"
        with open(File, 'a') as f:
            f.write(str(dev_loss) + "\n")

        modelName = time.asctime(time.localtime(time.time()))  # TIme
        modelName = modelName.split()  # Split
        modelName = modelName[1:4]  # Get only date and time info
        modelName[1] = modelName[1] + "_"
        modelName = "model/Model" + str(epoch) + "-" + ''.join(modelName) + ".h5"
        # print(modelName)

        if not Multi_GPU:
            model.save(modelName)
        else:
            m.save(modelName)

        if (dev_loss+train_loss) < best_dev:
            best_dev = dev_loss+train_loss
            # modelName = time.asctime(time.localtime(time.time())) # TIme
            # modelName = modelName.split() # Split
            # modelName = modelName[1:4] # Get only date and time info
            # modelName[1] = modelName[1]+"_"
            # modelName = "model/Model"+str(epoch)+"-"+''.join(modelName)+".h5"
            # # print(modelName)
            # model.save(modelName)
            # del model
            # model = load_model(modelName) ###
            # # Re-compile we initialize the optmize information (even using the same optimizer)
            # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) ###
            print("Just stored new opt model with (train+dev) loss:", str(best_dev))
    #
    # # t2 = time.clock()
    # # print(t2)
    # # print(t2-t1)

