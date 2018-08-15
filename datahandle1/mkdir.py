import os

if __name__ == '__main__':
    folder = 'data'
    class_name = 'class'
    for i in range(1,10):
        folderName = folder+ '\\' + class_name + str(i)
        os.makedirs(folderName)

        # print('Creating folders for class '+str(i)+' ...')
        # for j in range(1,601):
        #     sampleName = folderName + '\\sample' + str(j)
        #
        #     os.makedirs(sampleName,exist_ok=True)
        # print('Generating Done')

