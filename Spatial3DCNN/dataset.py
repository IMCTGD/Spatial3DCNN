import numpy as np
import os
import nibabel as nib
import csv
import pandas
import blockdataset
import numpy.random as random

# 制作txt文件
"""
输入cv0文件地址要test，
npy文件夹地址，
保存txt的地址的文件夹，
标签
result：“F:/block%d/.npy”
"""


def dataCv(cv0_test_path, npyDirPath, txtDir, label):
    train_data_txt = 'train_data.txt'
    test_data_txt = 'test_data.txt'
    train_label_txt = 'train_label.txt'
    test_label_txt = 'test_label.txt'
    test = []
    for line in open(cv0_test_path, 'r'):
        test.append(line[:-1])

    train_Filenames = []
    test_Filenames = []
    train_labels = []
    test_labels = []

    trainFilename_savePath = os.path.join(txtDir, train_data_txt)
    trainLabel_savePath = os.path.join(txtDir, train_label_txt)
    testFilename_savePath = os.path.join(txtDir, test_data_txt)
    testLabel_savePath = os.path.join(txtDir, test_label_txt)

    if not os.path.exists(txtDir):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(txtDir)

    for filename in os.listdir(npyDirPath):
        flag = 0
        fileStr = filename[8:18]
        for i in test:
            if fileStr == i:
                test_Filenames.append(filename)
                test_labels.append(label)
                flag = 1
                break
        if flag == 0:
            train_Filenames.append(filename)
            train_labels.append(label)
    test_data_f = open(testFilename_savePath, 'a')
    test_label_f = open(testLabel_savePath, 'a')
    # 按文件名
    npyDirPath_d = npyDirPath[:-1] + "%d"
    for test_filename in test_Filenames:
        test_filePath = os.path.join(npyDirPath_d, test_filename)
        test_data_f.write(test_filePath + '\n')
        test_label_f.write(str(label) + '\n')
    test_data_f.close()
    test_label_f.close()

    train_data_f = open(trainFilename_savePath, 'a')
    train_label_f = open(trainLabel_savePath, 'a')
    for train_filename in train_Filenames:
        train_filePath = os.path.join(npyDirPath_d, train_filename)
        train_data_f.write(train_filePath + '\n')
        train_label_f.write(str(label) + '\n')
    train_data_f.close()
    train_label_f.close()


# txt_dirPath文档文件夹地址，"F:/ADvsMCIc"
# txt里面的一条数据：“F:/AD/block%d/1.npy”
def dataset(txt_dirPath, blockNum):
    # train_data_path_txt = os.path.join(txt_dirPath, 'All_train.txt')
    # test_data_path_txt = os.path.join(txt_dirPath, 'All_test.txt')
    # train_label_path_txt = os.path.join(txt_dirPath, 'All_train_label.txt')
    # test_label_path_txt = os.path.join(txt_dirPath, 'All_test_label.txt')

    train_data_path_txt = os.path.join(txt_dirPath, 'train_data.txt')
    test_data_path_txt = os.path.join(txt_dirPath, 'test_data.txt')
    train_label_path_txt = os.path.join(txt_dirPath, 'train_label.txt')
    test_label_path_txt = os.path.join(txt_dirPath, 'test_label.txt')

    train_data_path_list = []
    train_data = []
    # 打开训练路径文件
    with open(train_data_path_txt, 'r') as f:
        for line in f.readlines():
            train_data_path_list.append(line.strip())
    # 读取nii文件存到train_data中
    for train_data_path in train_data_path_list:
        train_data_path = train_data_path % blockNum
        train_data.append(np.load(train_data_path))

    test_data_path_list = []
    test_data = []
    with open(test_data_path_txt, 'r') as f:
        for line in f.readlines():
            test_data_path_list.append(line.strip())
    for test_data_path in test_data_path_list:
        test_data_path = test_data_path % blockNum
        test_data.append(np.load(test_data_path))

    # 标签
    train_label = []
    with open(train_label_path_txt, 'r') as f:
        for line in f.readlines():
            train_label.append(int(line.strip()))

    test_label = []
    with open(test_label_path_txt, 'r') as f:
        for line in f.readlines():
            test_label.append(int(line.strip()))
    train_data = np.array(train_data)
    # print(train_data)
    # print(train_data.shape)
    train_data = np.expand_dims(train_data, axis=4)
    test_data = np.array(test_data)
    test_data = np.expand_dims(test_data, axis=4)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    # 数据增强至数量1：1
    # 修改的地方
    # for i in range(60):
    #     one_data = train_data[i, :, :, :, :]
    #     if random.uniform(0, 1) > 0.5:
    #         one_data = np.flip(one_data, axis=0)
    #     if random.uniform(0, 1) > 0.5:
    #         one_data = np.flip(one_data, axis=1)
    #     if random.uniform(0, 1) > 0.5:
    #         one_data = np.flip(one_data, axis=2)
    #     one_data = np.expand_dims(one_data, axis=0)
    #     train_data = np.concatenate((train_data, one_data), axis=0)
    #     train_label = np.append(train_label, 1)

    print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    dirPath = f'/home/caizhihong/T2ADNITrain/txt'
    for i in range(150):
        train_data, test_data, train_label, test_label = dataset(dirPath, i)
        print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
        print(train_label)
        print(test_label)
    # train_data, test_data, train_label, test_label = dataset("F:/ADvsMCIc", 80)
    # print(train_data)

    # if __name__ == "__main__":
    #     raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
    #                 'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
    #                 'age': [42, 52, 36, 24, 73],
    #                 'preTestScore': [4, 24, 31, 2, 3],
    #                 'postTestScore': [25, 94, 57, 62, 70]}
    #     df = pandas.DataFrame(raw_data, columns=['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
    #     print(df)
    #     df.to_csv('F:/log.csv')

        # data = [1, 2, 3, 4, 5]
        # dict = {'a':1,'b':2,'c':3}
        # label = [5,4,3,2,1]
        # with open('F:/log.csv', 'w') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['epoch', 'data', 'label'])
        #     for i in range(len(data)):
        #         writer.writerow([i, data[i], label[i]])
