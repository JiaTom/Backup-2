from dataIndexing import *
import cv2

if __name__ == '__main__':
    # sampleNum = 6  # 600
    # frameNum = 128
    # classNum = 3
    #
    # test_num = 2  # 100
    #
    # random.seed(1)
    # index = random.sample(range(1, sampleNum + 1), sampleNum)
    # train_index = index[:sampleNum - test_num]
    # test_index = index[sampleNum - test_num:]
    #
    # train, test = getClassSampleIndex(classNum, sampleNum, test_num)
    # print(train)
    # print(test)
    #
    # for index in range(0,len(test)):
    #     class_name = 'class'+ str(int(test[index][0]))
    #     sample_name= 's' + str(int(test[index][1]))
    #     # file = class_name + "/" + sample_name
    #     # print(file)
    #     for frame in range(1,frameNum+1):
    #         file = class_name + "/" + sample_name + 'frame' + str(frame)
    #         print(file)

    # img=cv2.imread("data2/class1/s1frame1.png")
    # print(img.shape)
    # img = img[:,:,0]
    # print(img.shape,np.sum(img))
    # img = np.reshape(img,(img.shape[0],img.shape[1],1))
    # print(img.shape)
    # img = cv2.resize(img,(299,299))
    # print(img.shape)

    t=np.random.random((600,128,299,299,1))