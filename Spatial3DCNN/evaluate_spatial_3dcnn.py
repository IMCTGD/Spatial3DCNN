import warnings
import tensorflow as tf

import utils
from spatial_3dcnn import Spatial3DCNN
import os
from matplotlib import pyplot as plt
import numpy as np
from dataset import dataset
import blockdataset
import csv

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def valid(category, blockNum, fold, result, data, model):

    print('begin train block %d' % blockNum)
    if category == 'ADvsHC':
        category1 = 'AD'
        category2 = 'HC'
    if category == 'MCIcvsHC':
        category1 = 'MCIc'
        category2 = 'HC'
    if category == 'MCIcvsMCInc':
        category1 = 'MCIc'
        category2 = 'MCInc'
    valid_FileDir1 = os.path.join(f'/home/caizhihong/Data/{data}/Fold/{fold}/block3D/valid', category1, 'block%d') % blockNum
    valid_FileDir2 = os.path.join(f'/home/caizhihong/Data/{data}/Fold/{fold}/block3D/valid', category2, 'block%d') % blockNum
    # 获取数据集
    valid_data_list, valid_label_list = utils.get_data(valid_FileDir1, valid_FileDir2)
    valid_data_list = np.array(valid_data_list)
    valid_label_list = np.array(valid_label_list, dtype=int)
    valid_label_list = np.eye(2)[valid_label_list]
    # print("valid_data_list", valid_data_list)
    # print("valid_label_list", valid_label_list)
    validdatatest = blockdataset.blockdataset(valid_data_list, valid_label_list, transform=None,
                                              status='test', batch_size=len(valid_data_list), shuffle=False)
    psn = Spatial3DCNN()
    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    psn.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss='binary_crossentropy',
                metrics=['accuracy'])
    # #######################加载模型##########################################################
    checkpoint_path = f'/home/caizhihong/CodeResult/{result}/Fold/{model}/checkpoint/{category}(1000)/psn{blockNum}.ckpt'
    psn.load_weights(checkpoint_path)

    loss, acc = psn.evaluate(x=valid_data_list, y=valid_label_list)
    return acc, loss


def test(category, blockNum, fold, result, model):

    print('begin train block %d' % blockNum)
    if category == 'ADvsHC':
        dirPath = f'/home/caizhihong/CodeResult/{result}/Fold/{model}/txt'
    train_data, test_data_list, train_label, test_label_list = dataset(dirPath, blockNum)
    test_label_list = np.eye(2)[test_label_list]
    psn = Spatial3DCNN()
    psn.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss='binary_crossentropy',
                metrics=['accuracy'])

    checkpoint_path = f'/home/caizhihong/CodeResult/{result}/Fold/{model}/checkpoint/{category}(1000)/psn{blockNum}.ckpt'
    psn.load_weights(checkpoint_path)

    loss, acc = psn.evaluate(x=test_data_list, y=test_label_list)
    return acc, loss


def main(category, fold='fold1', result='PETResult', model='fold1'):
    data = 'PETData'
    acc_list = []
    loss_list = []
    test_acc_list = []
    test_loss_list = []
    for i in range(150):
        acc, loss = valid(category, i, fold, result, data, model)
        test_acc, test_loss = test(category, i, fold, result, model)
        print("block:{}, acc:{}, loss:{}".format(i, acc, loss))
        loss_list.append(loss)
        acc_list.append(acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    with open(f'/home/caizhihong/CodeResult/{result}/Fold/{model}/log/ADvsHC/valid.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['block', 'val_acc', 'val_loss', 'test_acc', 'test_loss'])
        for i in range(150):
            writer.writerow([i, acc_list[i], loss_list[i], test_acc_list[i], test_loss_list[i]])


if __name__ == "__main__":
    # train('ADvsHC', 80)
    main('ADvsHC')
    print('************************************************')
    # max_N_Acc, max_N_index = main('MCIcvsHC')
    # max_N_Acc, max_N_index = main('MCIcvsMCInc')

