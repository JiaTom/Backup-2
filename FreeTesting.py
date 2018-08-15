import random
import numpy as np
import time
import math
import cv2
from dataIndexing import *

# data = np.zeros(shape=(500,128,299,299),dtype='float') # Maximum: 677 128 299 299
# t1 = time.clock()
# np.save('SaveTest0.npy',data)
# t2 = time.clock() # 811
# print(t2-t1)
# data2 = np.load('SaveTest0.npy') #617
# print(data2.shape)
#
# np.random.seed(1)
# data = np.random.random((10,128,299,299))
#
# print(data.shape)
#
# np.save('SaveTest.npy',data)
# data2 = np.load('SaveTest.npy')
# print(data2.shape)
#
# print(data2 == data)
#
# t1=time.clock()
# data = np.load('SaveTest0.npy')
# t2=time.clock()
# print(t2-t1)

def imgProcess(folder,file_index,frame_num=128,size=(299,299),channel=1):
    output = np.zeros(shape=(len(file_index),frame_num,size[0],size[1],channel))
    if not (channel == 1 or channel == 3):
        print('Channel must equal to 1 or 3')
        raise ValueError

    i = 0;
    for index in file_index:
        # print(index)
        for frame in range(1,frame_num+1):
            file = folder +"/class"+str(int(index[0]))+ "/s" + str(int(index[1])) + "frame" + str(int(frame))+".png"
            img = cv2.imread(file)
            img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
            if channel == 1:
                img = img[:,:,channel-1]
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
                # if i ==0:
                #     print(img.shape)
                #     print(output.shape)
            output[i,frame-1] = img
            # print(str(i), file)
            # print(frame)
        # print('end')
        i = i + 1
    return output

def indexsplit(index_len,interval_num):
    indexGroup = math.ceil(index_len/interval_num)
    output = np.zeros(shape=(indexGroup,2),dtype='int')
    for i in range(0,indexGroup-1):
        output[i,0]=i*interval_num
        output[i,1]=(i+1)*interval_num
    output[indexGroup-1,0] = (indexGroup-1)*interval_num
    output[indexGroup-1,1] = index_len-1
    return output


if __name__ == '__main__':
    sampleNum = 20
    frameNum = 128
    classNum = 10
    test_num = 2  # 100
    dev_num = 3

    train,dev,test = getClassSampleIndex(classNum, sampleNum, dev_num, test_num)

    print(train.shape,dev.shape,test.shape)
    # print(dev)
    # print(test)

    output_size = 60
    sample_index = indexsplit(len(train),output_size)
    print(sample_index)
    print(train[(sample_index[2,1])])

    for i in sample_index:
        data_index = train[i[0]:i[1]]
        print(data_index.shape)

        data = imgProcess('data2',file_index=data_index,frame_num=frameNum,size=(299,299))
        print(data.shape)





    # tt = imgProcess(folder='data2',file_index=dev,frame_num=frameNum,size=(299,299))
    # print(tt.shape)
    #
    # img_check = cv2.imread("data2/class1/s10frame1.png")
    # img_check = cv2.resize(img_check,(299,299),interpolation=cv2.INTER_LINEAR)
    # img_check = img_check[:,:,0]
    # img_check = np.reshape(img_check,(299,299,1))
    #
    # print(np.sum(img_check == tt[0,0]))




