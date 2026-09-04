import pandas as pd
import tensorflow as tf
from spatial_3dcnn import Spatial3DCNN
from meta_classifier import SpatialMetaClassifier
from ensemble_utils import separate_samples
import csv
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

modalData = 'ModalData'
modalResult = 'MRIResult'
data_type = 'MRI'
fold = 'fold1'
'''
该文件用于输出具体样本的预测结果
'''

def get_other_dataSet(dirPath_0, blockNum, AD_HC_list=['AD', 'HC']):
    block_str = 'block' + str(blockNum)
    nii_data_list = []
    nii_label_list = []
    filename_list =[]
    for category in AD_HC_list:
        dirPath_1 = os.path.join(dirPath_0, category, block_str)
        files = os.listdir(dirPath_1)
        files = sorted(files)
        for file in files:
            filename_list.append(file[:-4])
            nii_path = os.path.join(dirPath_1, file)
            nii_data_list.append(np.load(nii_path))
            if category == 'AD':
                nii_label_list.append(0)
            else:
                nii_label_list.append(1)

    #####变为numpy格式#######
    nii_data = np.array(nii_data_list)
    nii_data = np.expand_dims(nii_data, axis=4)
    nii_label = np.array(nii_label_list)
    nii_label = np.eye(2)[nii_label]
    return nii_data, nii_label, filename_list

# 用哪个fold的模型进行预测
acc_sort_path = f'/home/caizhihong/CodeResult/{modalResult}/ADvsHC/Fold/{fold}/log/ADvsHC/sort.csv'

# 读取大于0.75的block保存到blockNum_list
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

print(blockNum_list)

# 读取测试集数据并且获得中间层输出
test_midoutput = []

for blockNum in blockNum_list:
    # dirPath = f'/home/caizhihong/Data/{modalData}/AllData/{data_type}/block3D'
    # 你想预测的数据集
    dirPath = f'/home/caizhihong/Data/PETData/Fold/test/block3D'
    data, label, name = get_other_dataSet(dirPath, blockNum)

    # 加载基分类器模型，要与acc_sort_path的fold对应
    psn = Spatial3DCNN(return_features=True)
    psn.load_weights('/home/caizhihong/CodeResult/{}/ADvsHC/Fold/{}/checkpoint/ADvsHC(1000)/psn{}.ckpt'.format(modalResult, fold, blockNum))
    y, mid_output = psn.predict(x=data)
    test_midoutput.append(mid_output)

# 调整中间层输出的形状
test_samples = separate_samples(test_midoutput)
# 加载集成分类器模型
checkpoint_save_path = '/home/caizhihong/CodeResult/{}/ADvsHC/Fold/{}/checkpoint/feature_2.ckpt'.format(modalResult, fold)

block_Total_Num = test_samples.shape[2]
feature_Total_Num = test_samples.shape[1]
one_cnn = SpatialMetaClassifier(block_Total_Num, feature=feature_Total_Num)
one_cnn.compile(optimizer=tf.keras.optimizers.Adam(0.002), loss=tf.keras.losses.categorical_crossentropy,
                    metrics=['accuracy', tf.keras.metrics.AUC()])
one_cnn.load_weights(checkpoint_save_path)

# 获得准确率
loss, acc, auc = one_cnn.evaluate(test_samples, label)
print(acc)
# 预测
y = one_cnn.predict(x=test_samples)
# 转化为标签
y = np.argmax(y, axis=1)
label = np.argmax(label, axis=1)
print(y)
print(y.shape)
print(name)
print(len(name))
print(label)
print(len(label))
# 保存为csv文件
result_df = pd.DataFrame(
    {'SubjectID': name, 'prediction_label': y, 'true_label': label})
result_df.to_csv("/home/caizhihong/CodeResult/All_Pred_BaseNet_Ense/mri_test_predition.csv", index=False)
