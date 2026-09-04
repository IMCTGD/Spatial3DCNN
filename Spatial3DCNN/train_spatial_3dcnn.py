import tensorflow as tf

import utils
import os
from dataset import dataset
import blockdataset

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
from spatial_3dcnn import Spatial3DCNN


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)


def train(category, blockNum, fold, ResultFold, modalData, modalResult):

    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    batch_size = 32
    # 如果要fold，要在前添加fold1

    print('begin train block %d' % blockNum)
    if category == 'ADvsHC':
        dirPath = f'/home/caizhihong/CodeResult/{modalResult}/Fold/{ResultFold}/txt'
        category1 = 'AD'
        category2 = 'HC'

    train_data, test_data_list, train_label, test_label_list = dataset(dirPath, blockNum)
    train_label = np.eye(2)[train_label]
    # print("train_label:", train_label)
    valid_FileDir1 = os.path.join(f'/home/caizhihong/Data/{modalData}/Fold/{fold}/block3D/valid', category1, 'block%d') % blockNum
    valid_FileDir2 = os.path.join(f'/home/caizhihong/Data/{modalData}/Fold/{fold}/block3D/valid', category2, 'block%d') % blockNum
    # 获取数据集
    valid_data_list, valid_label_list = utils.get_data(valid_FileDir1, valid_FileDir2)
    valid_data_list = np.array(valid_data_list)
    valid_label_list = np.array(valid_label_list, dtype=int)
    valid_label_list = np.eye(2)[valid_label_list]

    traindataset = blockdataset.blockdataset(train_data, train_label,
                                             transform=[blockdataset.randomflip(), blockdataset.noisy(0.00),
                                                        blockdataset.randomflip180()],
                                             status='train', batch_size=batch_size, shuffle=True)

    train_gen = traindataset.data_generator()

    validdatatest = blockdataset.blockdataset(valid_data_list, valid_label_list, transform=None,
                                              status='test', batch_size=len(valid_data_list), shuffle=True)
    valid_gen = validdatatest.data_generator()

    psn = Spatial3DCNN()


    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    psn.compile(optimizer=tf.keras.optimizers.Adam(0.0000025), loss='binary_crossentropy',
                metrics=['accuracy'])

    checkpoint_save_path_dir = os.path.join(f'/home/caizhihong/CodeResult/{modalResult}/Fold/{ResultFold}/checkpoint/',
                                            category + '(1000)', "psn%d.ckpt")
    checkpoint_save_path = checkpoint_save_path_dir % blockNum

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = psn.fit(train_gen, steps_per_epoch=len(traindataset) // batch_size, epochs=1000,
                      callbacks=[earlystop, cp_callback], validation_data=valid_gen, validation_steps=1,
                      validation_freq=1,
                      workers=1,
                      use_multiprocessing=False, verbose=1)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')

    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt_path = os.path.join(f'/home/caizhihong/CodeResult/{modalResult}/Fold/{ResultFold}/plt',
                            "acc&loss%d.png") % blockNum

    plt.savefig(plt_path)
    plt.close()

    return val_acc


def main(category):
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    acc_list = []

    fold = 'fold1'
    ResultFold = 'fold1'
    modalData = 'PETData'
    modalResult = 'PETResult'
    print('------------------------', fold, '---------------------------')
    for i in range(150):
        acc = train(category, i, fold, ResultFold, modalData, modalResult)
        acc_list.append(max(acc))
    max_N_Acc = sorted(acc_list, reverse=True)
    max_N_index = sorted(range(len(max_N_Acc)), key=lambda x: max_N_Acc[x])

    log_dirPath = os.path.join(f"/home/caizhihong/CodeResult/{modalResult}/Fold/{ResultFold}/log", category)
    acc_path = os.path.join(log_dirPath, "acc.txt")
    index_path = os.path.join(log_dirPath, "index.txt")
    acc_f = open(acc_path, 'w')

    acc_f.write(str(max_N_Acc))
    acc_f.close()
    index_f = open(index_path, 'w')
    index_f.write(str(max_N_index))
    index_f.close()
    return max_N_Acc, max_N_index


if __name__ == "__main__":
    # train('ADvsHC', 80)
    max_N_Acc, max_N_index = main('ADvsHC')
    # max_N_Acc, max_N_index = main('MCIcvsHC')
    # max_N_Acc, max_N_index = main('MCIcvsMCInc')
    print('*****************************************************************')
# ############只要换fold序号就可以，会覆盖fold中的ADvsHC文件###########################
