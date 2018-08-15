import numpy as np
import time
import cv2

from dataIndexing import *

## Test Version
# def imgProcess(folder,file_index,frame_num=128,size=(299,299),channel=1):
#     output = np.zeros(shape=(len(file_index),frame_num,size[0],size[1],channel))
#     if not (channel == 1 or channel == 3):
#         print('Channel must equal to 1 or 3')
#         raise ValueError
#
#     i = 0;
#     for index in file_index:
#         # print(index)
#         for frame in range(1,frame_num+1):
#             file = folder +"/class"+str(int(index[0]))+ "/s" + str(int(index[1])) + "frame" + str(int(frame))+".png"
#             img = cv2.imread(file)
#             img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
#             if channel == 1:
#                 img = img[:,:,channel-1]
#                 img = np.reshape(img,(img.shape[0],img.shape[1],1))
#                 # if i ==0:
#                 #     print(img.shape)
#                 #     print(output.shape)
#             output[i,frame-1] = img
#             # print(str(i), file)
#             # print(frame)
#         # print('end')
#         i = i + 1
#     return output
#
# tt = imgProcess(folder='data2',file_index=dev,frame_num=frameNum,size=(299,299))
# print(tt.shape)
#
# img_check = cv2.imread("data2/class1/s10frame1.png")
# img_check = cv2.resize(img_check,(299,299),interpolation=cv2.INTER_LINEAR)
# img_check = img_check[:,:,0]
# img_check = np.reshape(img_check,(299,299,1))
#
# print(np.sum(img_check == tt[0,0]))


def imgProcess(folder,file_index,frame_num=128,size=(299,299),channel=1,store=None):
    if not (channel == 1 or channel == 3):
        print('Channel must equal to 1 or 3')
        raise ValueError

    if not store:
        output = np.zeros(shape=(len(file_index), frame_num, size[0], size[1], channel))
    if store: # dtype = int if checking equality
        output = np.memmap(store, dtype='float', mode='w+', shape=(len(file_index), frame_num,
                                                                  size[0],size[1],channel))

    i = 0;
    for index in file_index:
        print(index)
        for frame in range(1,frame_num+1):
            file = folder +"/class"+str(int(index[0]))+ "/s" + str(int(index[1])) + "frame" + str(int(frame))+".png"
            # print(file)
            img = cv2.imread(file)
            # print(img.shape)
            img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
            if channel == 1:
                img = img[:,:,channel-1]
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
            output[i,frame-1] = img
        # print('end')
        i = i + 1

    if not store:
        return output
    else:
        del output
        ts=time.clock() # dtype = int if checking equality
        output = np.memmap(store, dtype='float', mode='r', shape=(len(file_index), frame_num,
                                                                  size[0],size[1],channel))
        te=time.clock()
        string = 'Loading '+store+' using:' + str(int(te-ts))
        print(string)
        return output

def indexsplit(index_len,interval_num):
    indexGroup = math.ceil(index_len/interval_num)
    output = np.zeros(shape=(indexGroup,2),dtype='int')
    for i in range(0,indexGroup-1):
        output[i,0]=i*interval_num
        output[i,1]=(i+1)*interval_num
    output[indexGroup-1,0] = (indexGroup-1)*interval_num
    output[indexGroup-1,1] = index_len
    return output


if __name__ == '__main__':
    sampleNum = 1000  # 600
    frameNum = 128
    classNum = 3
    dev_num = 150  # 150
    test_num = 150  # 50

    train, dev, test = getClassSampleIndex(classNum, sampleNum, dev_num, test_num)
    np.random.seed(1)
    np.random.shuffle(train)
    np.random.seed(1)
    np.random.shuffle(dev)
    np.random.seed(1)
    np.random.shuffle(test)
    print(train.shape, dev.shape, test.shape)
    print(len(train[6650:7000]))

    foldsize = 350
    train_index = indexsplit(len(train), foldsize)
    dev_index = indexsplit(len(dev),foldsize)
    test_index = indexsplit(len(test),foldsize)

    print(train_index.shape,dev_index.shape)

    #############################################################################################################

    for i in range(0,len(train_index)):
        storeFile = 'SR6/train'+str(i+1)+".dat" #
        # storeFile = 'Dat299/train' + str(i + 1) + ".dat"  #
        print(storeFile)
        ind = train_index[i] #
        index = train[ind[0]:ind[1]] #
        print(ind,index.shape)

        t1 = time.clock()
        data = imgProcess("data6", file_index=index, frame_num=frameNum, size=(64, 64), store=storeFile)
        # data = imgProcess("data", file_index=index, frame_num=frameNum, size=(299, 299), store=storeFile)
        t2 = time.clock()
        print(t2 - t1)
        del data

    for i in range(0,len(dev_index)):
        storeFile = 'SR6/dev'+str(i+1)+".dat" #
        # storeFile = 'Dat299/dev' + str(i + 1) + ".dat"
        print(storeFile)
        ind = dev_index[i] #
        index = dev[ind[0]:ind[1]] #
        print(ind,index.shape)

        t1 = time.clock()
        data = imgProcess("data6", file_index=index, frame_num=frameNum, size=(64, 64), store=storeFile)
        # data = imgProcess("data", file_index=index, frame_num=frameNum, size=(299, 299), store=storeFile)
        t2 = time.clock()
        print(t2 - t1)
        del data

    for i in range(0,len(test_index)): #
        storeFile = 'SR6/test'+str(i+1)+".dat" #
        # storeFile = 'Dat299/test'+str(i+1)+".dat"
        print(storeFile)
        ind = test_index[i] #
        index = test[ind[0]:ind[1]] #
        print(ind,index.shape)

        t1 = time.clock()
        data = imgProcess("data6", file_index=index, frame_num=frameNum, size=(64, 64), store=storeFile)
        # data = imgProcess("data", file_index=index, frame_num=frameNum, size=(299, 299), store=storeFile)
        t2 = time.clock()
        print(t2 - t1)
        del data

    #############################################################################################################


    #Testing for 1 folder and check equal  (set foldsize = 50 and np.memap dtype = 'float')
    # testFold=18
    # ind = train_index[testFold]
    # print(ind)
    # index = train[ind[0]:ind[1]]
    # print(index.shape,index[250],train[250])
    #
    # t1 = time.clock()
    # data1 = imgProcess("data/data", file_index=index, frame_num=frameNum,
    #                    size=(64, 64),store='EqualTest/train18.dat') ##### Modified
    # filename = "EqualTest/train"+str(int(testFold))+".dat"
    # print(filename)
    # # Notice!!!: When Checking equality, dtype including imgProcess must be set as 'int'
    # data2 = np.memmap(filename, dtype='int', mode='r', shape=(350, 128, 64, 64, 1))
    # t2 = time.clock()
    # print(t2-t1)
    # data3 = imgProcess("data/data", file_index=index, frame_num=frameNum, size=(64, 64))
    # print(np.sum(data1==data2))
    # print(np.sum(data2==data3))


    '''Construct Output (class) data'''
    from keras.utils import to_categorical

    for i in range(0,len(test_index)): #
        storeFile = "SR6/outputDAT/testCls"+str(i+1)+".npy" #
        print(storeFile)
        ind = test_index[i] #
        index = test[ind[0]:ind[1]] #
        print(ind,index.shape)

        cls = [[int(i[0]) - 1] for i in index]
        cls = to_categorical(cls, classNum)
        print(cls.shape)
        output = cls.copy()
        # print(cls)
        # print(output)
        np.save(storeFile, output)
        del output

    for i in range(0,len(dev_index)): #
        storeFile = "SR6/outputDAT/devCls"+str(i+1)+".npy" #
        print(storeFile)
        ind = dev_index[i] #
        index = dev[ind[0]:ind[1]] #
        print(ind,index.shape)

        cls = [[int(i[0]) - 1] for i in index]
        cls = to_categorical(cls, classNum)
        print(cls.shape)
        output = cls.copy()
        np.save(storeFile, output)
        del output

    for i in range(0,len(train_index)): #
        storeFile = "SR6/outputDAT/trainCls"+str(i+1)+".npy" #
        print(storeFile)
        ind = train_index[i] #
        index = train[ind[0]:ind[1]] #
        print(ind,index.shape)

        cls = [[int(i[0]) - 1] for i in index]
        cls = to_categorical(cls, classNum)
        print(cls.shape)
        output = cls.copy()
        np.save(storeFile, output)
        del output









    














    # arr = np.arange(9).reshape((3, 3))
    #
    # np.random.seed(1)
    # np.random.shuffle(arr)
    # arr1 = arr.copy()
    #
    # np.random.seed(2)
    # np.random.shuffle(arr)
    # arr2 = arr.copy()
    #
    # np.random.seed(3)
    # np.random.shuffle(arr)
    # arr3 = arr.copy()
    #
    # arr = np.arange(9).reshape((3, 3))
    #
    # print(arr)
    # print(arr1)
    # print(arr2)
    # print(arr3)
    #
    # tt = np.zeros(shape=(4,3,3),dtype=float) # 4 is the total sample number
    # tt[0] = arr
    # tt[1] = arr1
    # tt[2] = arr2
    # tt[3] = arr3
    # print(tt,tt.dtype)


