import numpy as np
import random
import math

def getTrainTestIndex(sampleNum,test_num,dev_num,currentClass):
    random.seed(currentClass)
    index = random.sample(range(1,sampleNum+1), sampleNum)
    train_index = index[:sampleNum - test_num - dev_num]
    dev_index = index[sampleNum - test_num - dev_num:sampleNum - test_num]
    test_index = index[sampleNum - test_num:]

    train = np.zeros(shape=(len(train_index),2))
    test = np.zeros(shape=(len(test_index),2))
    dev = np.zeros(shape=(len(dev_index),2))
    for i in range(len(train_index)):
        train[i,0] = currentClass
        train[i,1] = train_index[i]
    for i in range(len(test_index)):
        test[i,0] = currentClass
        test[i,1] = test_index[i]
    for i in range(len(dev_index)):
        dev[i,0] = currentClass
        dev[i,1] = dev_index[i]

    return train,dev,test

def getClassSampleIndex(classNum,sampleNum,dev_num=None,test_num=None,dev_need=True):
    if not dev_num:
        print('Must specified dev number or proportion, no mater whther dev set is required')
        raise ValueError
    if not test_num:
        print('Must specified test number or proportion')
        raise ValueError
    else:
        testtype = type(test_num).__name__
        devtype = type(dev_num).__name__
        if (not testtype=='int') and (not testtype=='float'):
            print('Invalid test number entered')
            raise ValueError
        if (not devtype=='int') and (not devtype=='float'):
            print('Invalid dev number entered')
            raise ValueError

    if testtype == 'float':
        test_num = math.ceil(sampleNum*test_num)
    if devtype == 'float':
        dev_num = math.ceil(sampleNum*dev_num)
    if not dev_need:
        test_num = test_num / 2
        dev_num = test_num

    train = None
    test = None
    dev = None
    for i in range(1,classNum+1):
        train_index,dev_index,test_index = getTrainTestIndex(sampleNum,test_num,dev_num,currentClass=i)
        if type(train).__name__ == 'NoneType':
            train = train_index.copy()
        else:
            train = np.concatenate((train,train_index),axis=0)

        if type(test).__name__ == 'NoneType':
            test = test_index.copy()
        else:
            test = np.concatenate((test,test_index),axis=0)

        if type(dev).__name__ == 'NoneType':
            dev = dev_index.copy()
        else:
            dev = np.concatenate((dev,dev_index),axis=0)

    if not dev_need:
        test = np.append(dev,test,axis=0)
        return train,test
    else:
        return train,dev,test

if __name__ == '__main__':
    # Testing: 3 classes with 6 samples. Each class take 2 samples in test and rest in training
    sampleNum = 20
    frameNum = 128
    classNum = 10
    test_num = 2  # 100
    dev_num = 3

    train, dev, test = getClassSampleIndex(classNum, sampleNum, dev_num, test_num)

    print(train.shape, dev.shape, test.shape)

    #########################################
    sampleNum = 600  # 600
    frameNum = 128
    classNum = 21
    test_num = 100 # 100

    train, test = getClassSampleIndex(classNum, sampleNum, test_num)
    print(train.shape,test.shape)



