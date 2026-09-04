import tensorflow as tf
import numpy as np
import csv
import os
from shutil import copy
from collections import defaultdict

from data_utils import get_other_dataset
from spatial_3dcnn import Spatial3DCNN
from meta_classifier import SpatialMetaClassifier
from ensemble_utils import train_gen, separate_samples, softmax, evaluate_feature, mcc_compute
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_mid_output(type_model, blockNum, fold, modalResult, modalData):
    print('begin get block mid_output %d' % blockNum)
    if type_model == 'train':
        dirPath = f'/home/caizhihong/Data/{modalData}/Fold/{fold}/block3D/train'
        data, label = get_other_dataset(dirPath, blockNum)
        data, label = train_gen(data, label)
    elif type_model == 'valid':
        dirPath = f'/home/caizhihong/Data/{modalData}/Fold/{fold}/block3D/valid'
        data, label = get_other_dataset(dirPath, blockNum)
    else:
        dirPath = f'/home/caizhihong/Data/{modalData}/Fold/test/block3D'
        # dirPath = f'/home/caizhihong/Data/{modalData}/Fold/test/block3D'
        data, label = get_other_dataset(dirPath, blockNum)

    psn = Spatial3DCNN(return_features=True)
    psn.load_weights('/home/caizhihong/CodeResult/{}/Fold/{}/checkpoint/ADvsHC(1000)/psn{}.ckpt'.format(modalResult, fold, blockNum))
    y, mid_output = psn.predict(x=data)
    # prediction_result_binary = np.argmax(y, axis=1)
    # print(y)
    # print(prediction_result_binary)
    return mid_output, label


def get_sample(blockNum_list, fold, modalResult, modalData):
    print(blockNum_list)
    mid_outputs = []
    valid_mid_outputs = []
    test_mid_outputs = []
    # 获取全部块中间输出
    for blockNum in blockNum_list:
        mid_output, train_label = get_mid_output('train', blockNum, fold, modalResult, modalData)
        valid_mid_output, valid_label = get_mid_output('valid', blockNum, fold, modalResult, modalData)
        mid_outputs.append(mid_output)
        valid_mid_outputs.append(valid_mid_output)
        test_mid_output, test_label = get_mid_output('test', blockNum, fold, modalResult, modalData)
        test_mid_outputs.append(test_mid_output)
    # 分样本排列数据
    samples = separate_samples(mid_outputs)  # (239,feature,block_num)
    valid_samples = separate_samples(valid_mid_outputs)
    test_samples = separate_samples(test_mid_outputs)
    return samples, valid_samples, test_samples, train_label, valid_label, test_label


def other_train_feature(samples, valid_samples, train_label, valid_label, num_i, fold, modalResult):
    block_Total_Num = samples.shape[2]
    feature_Total_Num = samples.shape[1]
    one_cnn = SpatialMetaClassifier(block_Total_Num, feature=feature_Total_Num)
    checkpoint_save_path = '/home/caizhihong/CodeResult/{}/Fold/{}/checkpoint/feature_{}.ckpt'.format(modalResult, fold, num_i)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    one_cnn.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss=tf.keras.losses.categorical_crossentropy,
                    metrics=['accuracy', tf.keras.metrics.AUC()])
    one_cnn.fit(x=samples, y=train_label, batch_size=16, epochs=100, callbacks=[cp_callback],
                validation_data=(valid_samples, valid_label))
    one_cnn.summary()
    one_cnn.load_weights(checkpoint_save_path)
    restore_v_target_name = 'spatial_weights'
    for v in one_cnn.trainable_variables:
        print(v.name)
        if restore_v_target_name in v.name:
            weight_shape_train = v.shape  # shape=[5,1,1]
            weight_train = v.numpy()
            break
    # return weight, weight_shape, one_cnn
    return weight_train, weight_shape_train, one_cnn


def other_secondOp_main(dataset, fold, modalResult, modalData):
    acc_sort_path = f'/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/log/ADvsHC/sort.csv'
    best_acc_dirPath_0 = f'/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/best_acc'

    with open(acc_sort_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        raw_blockNum_list = [row[0] for row in reader]
    with open(acc_sort_path, 'r') as csvfile:
        reader1 = csv.reader(csvfile)
        blockAcc_list = [row[1] for row in reader1]  #######修改#######
    raw_blockNum_list = raw_blockNum_list[1:]
    blockAcc_list = blockAcc_list[1:]
    for i in range(len(raw_blockNum_list)):
        raw_blockNum_list[i] = int(raw_blockNum_list[i])
        blockAcc_list[i] = float(blockAcc_list[i])
    blockNum_list = []
    for i in range(len(blockAcc_list)):
        if blockAcc_list[i] >= 0.75:
            blockNum_list.append(raw_blockNum_list[i])

    samples, valid_samples, test_samples, train_label, valid_label, test_label = get_sample(blockNum_list, fold, modalResult, modalData)

    num_list = [i for i in range(1, 101)]
    for num_i in num_list:
        weight, weight_shape, one_cnn = other_train_feature(samples, valid_samples, train_label, valid_label, num_i, fold, modalResult)
        weight = np.reshape(weight, -1)
        print('weight:', weight)
        weight_abs = np.abs(weight)
        print('weight(abs)', weight_abs)

        #########保存的地址##########
        best_acc_dirPath = os.path.join(best_acc_dirPath_0, '{}'.format(num_i))
        if not os.path.exists(best_acc_dirPath):
            os.makedirs(best_acc_dirPath)
        best_acc_path = os.path.join(best_acc_dirPath, 'weight.txt')
        best_valid_path = os.path.join(best_acc_dirPath, 'feature_valid.csv')
        best_test_path = os.path.join(best_acc_dirPath, 'feature_test.csv')
        plt_path = os.path.join(best_acc_dirPath, "top_N_acc.png")

        with open(best_acc_path, 'w', encoding='UTF-8') as f:
            f.write('weight:')
            f.write(str(weight))
            f.write('\n')
            f.write('weight(abs)' + str(weight_abs))
        weight_softmax = softmax(weight_abs)
        # 获取排列后的索引大小如[1,2,6,4,3,5]-->[0,1,4,3,5,2]
        weight_arg = np.argsort(weight_softmax)
        print(weight_arg)

        block_dict = dict(zip(blockNum_list, weight_softmax))
        block_dict = dict(sorted(block_dict.items(), key=lambda kv: (kv[1], kv[0])))
        block_num_list = []
        for block_num in block_dict.keys():
            block_num_list.append(int(block_num))
        # 得到新的样本数据集
        valid_acc_list, valid_auc_list, valid_mcc_list = [], [], []
        test_acc_list, test_auc_list, test_mcc_list = [], [], []
        for i in range(len(blockNum_list)):
            # 验证集MCC
            valid_acc, valid_loss, valid_auc, valid_y_pred = evaluate_feature(valid_samples, valid_label, one_cnn,
                                                                              weight)
            valid_mcc = mcc_compute(valid_label, valid_y_pred)
            valid_acc_list.append(valid_acc)
            valid_auc_list.append(valid_auc)
            valid_mcc_list.append(valid_mcc)
            print(valid_acc, valid_loss)
            print('auc:', valid_auc)
            print('muc:', valid_mcc)

            # 测试集
            test_acc, test_loss, test_auc, test_y_pred = evaluate_feature(test_samples, test_label, one_cnn,
                                                                          weight)
            test_mcc = mcc_compute(test_label, test_y_pred)
            test_acc_list.append(test_acc)
            test_auc_list.append(test_auc)
            test_mcc_list.append(test_mcc)
            print(test_acc, test_loss)
            print('auc:', test_auc)
            print('muc:', test_mcc)

            weight[weight_arg[i]] = 0

        with open(best_valid_path, 'w', newline='') as csvfile:
            # ##################################cv_1######################################
            writer = csv.writer(csvfile)
            writer.writerow(['block', 'val_acc', 'auc', 'mcc'])
            for i in range(len(blockNum_list)):
                writer.writerow([len(blockNum_list) - i, valid_acc_list[i], valid_auc_list[i], valid_mcc_list[i]])
        with open(best_test_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['block', 'test_acc', 'test_auc', 'test_mcc'])
            for i in range(len(blockNum_list)):
                writer.writerow([len(blockNum_list) - i, test_acc_list[i], test_auc_list[i], test_mcc_list[i]])

        print("########################开始画图############################################")
        print(blockNum_list)
        len_block_list = len(blockNum_list)
        top_N = [len_block_list - i for i in range(0, len(blockNum_list))]
        # 画验证集
        plt.subplot(1, 2, 1)
        plt.plot(top_N, valid_acc_list, label='ACC')
        plt.plot(top_N, valid_auc_list, label='AUC')
        plt.plot(top_N, valid_mcc_list, label='MCC')
        plt.title('valid_set')
        plt.xlabel("top_N")
        valid_acc_max_index = np.argmax(valid_acc_list)
        plt.annotate(text="%.2f" % np.max(valid_acc_list),
                     xy=(valid_acc_max_index, valid_acc_list[valid_acc_max_index]),
                     xytext=(valid_acc_max_index, valid_acc_list[valid_acc_max_index]))
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(top_N, test_acc_list, label='ACC')
        plt.plot(top_N, test_auc_list, label='AUC')
        plt.plot(top_N, test_mcc_list, label='MCC')
        plt.title('test_set')
        plt.xlabel("top_N")

        plt.legend()

        plt.savefig(plt_path)
        test_acc_max_index = np.argmax(test_acc_list)
        plt.annotate(text="%.2f" % np.max(test_acc_list), xy=(test_acc_max_index, test_acc_list[test_acc_max_index]),
                     xytext=(test_acc_max_index, test_acc_list[test_acc_max_index]))
        # plt.show()
        plt.close()
        tf.keras.backend.clear_session()


def select_top10_validACC(dirPath):
    acc_dict = {}
    for i in range(1, 101):
        path = os.path.join(dirPath, str(i), 'feature_test.csv')
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'block':
                    continue
                acc_dict[i] = float(row[1])
                break
    acc_dict = dict(sorted(acc_dict.items(), key=lambda item: item[1], reverse=True))
    save_list = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
    num = 0
    for select_i in acc_dict.keys():
        if num >= 10:
            break
        new_path_0 = os.path.join(dirPath, str(select_i), 'feature_test.csv')
        new_path_1 = os.path.join(dirPath, str(select_i), 'feature_valid.csv')
        new_path_2 = os.path.join(dirPath, str(select_i), 'top_N_acc.png')
        new_path_3 = os.path.join(dirPath, str(select_i), 'weight.txt')
        save_path = os.path.join(dirPath, str(save_list[num]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        copy(new_path_0, save_path)
        copy(new_path_1, save_path)
        copy(new_path_2, save_path)
        copy(new_path_3, save_path)
        num += 1


def get_acc_mcc_auc(dirPath, fold, modalResult):
    #########改为大于70的地址##############
    #
    # example_list = [str(i) for i in range(1, 101)]
    example_list = ['1st', '2nd', '3rd', '4th', '5th']
    valid_acc_volumn = defaultdict(list)
    test_acc_volumn = defaultdict(list)
    valid_auc_volumn = defaultdict(list)
    test_auc_volumn = defaultdict(list)
    valid_mcc_volumn = defaultdict(list)
    test_mcc_volumn = defaultdict(list)

    for order_num in example_list:
        filePath = os.path.join(dirPath, 'best_acc', order_num)
        valid_path = os.path.join(filePath, 'feature_valid.csv')
        test_path = os.path.join(filePath, 'feature_test.csv')
        print(test_path)
        with open(valid_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:

                if row[0] == 'block':
                    continue
                valid_acc_volumn[row[0]].append(float(row[1]))
                valid_auc_volumn[row[0]].append(float(row[2]))
                valid_mcc_volumn[row[0]].append(float(row[3]))

        with open(test_path, 'r', newline='') as csvfile1:
            reader = csv.reader(csvfile1)
            for row in reader:
                if row[0] == 'block':
                    continue
                test_acc_volumn[row[0]].append(float(row[1]))
                test_auc_volumn[row[0]].append(float(row[2]))
                test_mcc_volumn[row[0]].append(float(row[3]))

    print(valid_acc_volumn)
    print('test_acc_volumn:', test_acc_volumn)

    valid_acc_volumn_mean, test_acc_volumn_mean = {}, {}
    valid_acc_volumn_std, test_acc_volumn_std = {}, {}

    valid_auc_volumn_mean, test_auc_volumn_mean = {}, {}
    valid_auc_volumn_std, test_auc_volumn_std = {}, {}

    valid_mcc_volumn_mean, test_mcc_volumn_mean = {}, {}
    valid_mcc_volumn_std, test_mcc_volumn_std = {}, {}
    slice_length = len(valid_acc_volumn)
    for i in range(1, slice_length):
        valid_acc_volumn_mean[str(i)] = np.mean(valid_acc_volumn[str(i)])
        test_acc_volumn_mean[str(i)] = np.mean(test_acc_volumn[str(i)])
        valid_acc_volumn_std[str(i)] = np.std(valid_acc_volumn[str(i)])
        test_acc_volumn_std[str(i)] = np.std(test_acc_volumn[str(i)])

        ##########auc#########
        valid_auc_volumn_mean[str(i)] = np.mean(valid_auc_volumn[str(i)])
        test_auc_volumn_mean[str(i)] = np.mean(test_auc_volumn[str(i)])
        valid_auc_volumn_std[str(i)] = np.std(valid_auc_volumn[str(i)])
        test_auc_volumn_std[str(i)] = np.std(test_auc_volumn[str(i)])

        ########MCC###########
        valid_mcc_volumn_mean[str(i)] = np.mean(valid_mcc_volumn[str(i)])
        test_mcc_volumn_mean[str(i)] = np.mean(test_mcc_volumn[str(i)])
        valid_mcc_volumn_std[str(i)] = np.std(valid_mcc_volumn[str(i)])
        test_mcc_volumn_std[str(i)] = np.std(test_mcc_volumn[str(i)])
    print(valid_acc_volumn_mean)

    ########地址改为大于70########
    save_path = f'/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/log/auc_acc_mcc.csv'

    with open(save_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['AUC', 'ACC', 'MCC', 'val_auc_std', 'val_acc_std', 'val_mcc_std', 'AUC', 'ACC', 'MCC', 'test_auc_std',
             'test_acc_std', 'test_mcc_std'])
        for write_i in valid_auc_volumn_mean.keys():
            row = [valid_auc_volumn_mean[write_i], valid_acc_volumn_mean[write_i], valid_mcc_volumn_mean[write_i],
                   valid_auc_volumn_std[write_i], valid_acc_volumn_std[write_i], valid_mcc_volumn_std[write_i],
                   test_auc_volumn_mean[write_i], test_acc_volumn_mean[write_i], test_mcc_volumn_mean[write_i],
                   test_auc_volumn_std[write_i], test_acc_volumn_std[write_i], test_mcc_volumn_std[write_i]]
            writer.writerow(row)

if __name__ == '__main__':
    dataset = 'oasis'
    fold = 'fold1'
    modalResult = 'PETResult'
    modalData = 'PETData'
    other_secondOp_main(dataset, fold, modalResult, modalData)
    dirPath = f'/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/best_acc'
    select_top10_validACC(dirPath)
    dirPath1 = f'/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}'
    get_acc_mcc_auc(dirPath1, fold, modalResult)
