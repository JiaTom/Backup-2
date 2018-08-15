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

    # Create New Model
    model = create_model(128,64,64,1)
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01,momentum=0.9),metrics=['accuracy'])
    print(model.summary())

    # # Load Model
    # modelFile = "SGD/model/Model7-Jun26_19:55:46.h5"
    # model = load_model(modelFile)
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.8), metrics=['accuracy'])
    # print(model.summary())
    #
    #
    # xFile = "data/train"
    # yFile = "data/outputDAT/trainCls"
    # xDevFile = "data/dev"
    # yDevFile = "data/outputDAT/devCls"
    #
    #
    #
    # epoch_num = 300 #100
    # best_dev = 10
    # # t1 = time.clock()
    # for epoch in range(epoch_num):
    #     print("############### Epoch",str(epoch+1)," #################")
    #
    #     train_loss = []
    #     dev_loss = []
    #
    #     print("______________Training_________________")
    #     for i in range(20): ### 20
    #         x_train_file = xFile + str(i+1)+".dat"
    #         # print("Loading x training data from",x_train_file)
    #         x_train_load = np.memmap(x_train_file, dtype='float', mode='r', shape=(350, 128, 64, 64, 1))
    #         x_train = x_train_load.copy()
    #         del x_train_load
    #         print(x_train.shape)
    #
    #         y_train_file = yFile + str(i+1)+".npy"
    #         # print("Loading y training data from", y_train_file)
    #         y_train_load = np.load(y_train_file)
    #         y_train = y_train_load.copy()
    #         del y_train_load
    #         # print(y_train.shape)
    #
    #         # TODO: Delete
    #         # x_train = x_train[:20] ###
    #         # y_train = y_train[:20] ###
    #         print("Training model on training data",str(i+1)+"/20 ...")
    #         model.fit(x_train, y_train, epochs=1, batch_size=3, verbose=2)  # Maximum batch size: 3
    #         print("Evaluating model on training data", str(i + 1) + "/20 ...")
    #         eval = model.evaluate(x_train,y_train,batch_size=1,verbose=2)
    #         train_loss.append(eval[0])
    #         del x_train
    #         del y_train
    #
    #     # print(train_loss)
    #     train_loss = np.average(train_loss)
    #     print(train_loss)
    #     # Write loss result to file
    #     File = "history/TrainLoss.txt"
    #     with open(File, 'a') as f:
    #         f.write(str(train_loss) + "\n")
    #
    #     print("______________Evaluating_________________")
    #
    #     for i in range(5): ### 5 # 3 for final
    #         x_dev_file = xDevFile + str(i + 1) + ".dat"
    #         # print("Loading x developing data from", x_dev_file)
    #         sample_num = 350
    #         if i+1 == 5:
    #             sample_num = 100 # 300 for final
    #         x_dev_load = np.memmap(x_dev_file, dtype='float', mode='r', shape=(sample_num, 128, 64, 64, 1))
    #         x_dev = x_dev_load.copy()
    #         del x_dev_load
    #         print(x_dev.shape)
    #
    #         y_dev_file = yDevFile + str(i + 1) + ".npy"
    #         # print("Loading y developing data from", y_dev_file)
    #         y_dev_load = np.load(y_dev_file)
    #         y_dev = y_dev_load.copy()
    #         del y_dev_load
    #         # print(y_dev.shape)
    #
    #         # TODO: Delete
    #         # x_dev = x_dev[:10]  ###
    #         # y_dev = y_dev[:10]  ###
    #         print("Evaluating model on developing data", str(i + 1) + "/5 ...")
    #         eval = model.evaluate(x_dev,y_dev,batch_size=1,verbose=2)
    #         dev_loss.append(eval[0])
    #         del x_dev
    #         del y_dev
    #
    #     # print(dev_loss)
    #     # print(len(dev_loss))
    #     # dev_loss = np.average(dev_loss)
    #     tmp=0
    #     for i in range(len(dev_loss)):
    #         if i+1 < 5:
    #             tmp += (350/1500)*dev_loss[i]
    #         else:
    #             tmp += (100/1500)*dev_loss[i]
    #     dev_loss = tmp
    #     print(dev_loss)
    #     # write loss result to file
    #     File = "history/DevLoss.txt"
    #     with open(File, 'a') as f:
    #         f.write(str(dev_loss) + "\n")
    #
    #     modelName = time.asctime(time.localtime(time.time()))  # TIme
    #     modelName = modelName.split()  # Split
    #     modelName = modelName[1:4]  # Get only date and time info
    #     modelName[1] = modelName[1] + "_"
    #     modelName = "model/Model" + str(epoch) + "-" + ''.join(modelName) + ".h5"
    #     # print(modelName)
    #     model.save(modelName)
    #
    #     if (dev_loss+train_loss) < best_dev:
    #         best_dev = dev_loss+train_loss
    #         # modelName = time.asctime(time.localtime(time.time())) # TIme
    #         # modelName = modelName.split() # Split
    #         # modelName = modelName[1:4] # Get only date and time info
    #         # modelName[1] = modelName[1]+"_"
    #         # modelName = "model/Model"+str(epoch)+"-"+''.join(modelName)+".h5"
    #         # # print(modelName)
    #         # model.save(modelName)
    #         # del model
    #         # model = load_model(modelName) ###
    #         # # Re-compile we initialize the optmize information (even using the same optimizer)
    #         # model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) ###
    #         print("Just stored new opt model with loss:", str(best_dev))
    #
    # # t2 = time.clock()
    # # print(t2-t1)

