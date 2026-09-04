'''
该文件的作用是分割数据集，将数据集分割为训练集和测试集，并将其保存为文件，输出到txt文件。
'''
import os
import random

modalData = 'ModalData'
modalResult = 'ModalResult'
fold = 'fold1'
test_name = 'test'
type_data = 'PET'
# 数据集目录
ad_dataset_dir = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{fold}/train/AD"
cn_dataset_dir = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{fold}/train/HC"
ad_block_dir = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{fold}/block3D/train/AD"
cn_block_dir = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{fold}/block3D/train/HC"

# ad_dataset_dir = f"/home/caizhihong/Data/{modalData}/Fold/{fold}/valid/AD"
# cn_dataset_dir = f"/home/caizhihong/Data/{modalData}/Fold/{fold}/valid/HC"
# ad_block_dir = f"/home/caizhihong/Data/{modalData}/Fold/{fold}/block3D/valid/AD"
# cn_block_dir = f"/home/caizhihong/Data/{modalData}/Fold/{fold}/block3D/valid/HC"

test_ad = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{test_name}/AD"
test_cn = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{test_name}/HC"
test_ad_block = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{test_name}/block3D/AD"
test_cn_block = f"/home/caizhihong/Data/{modalData}/Fold/MutilBranch/{type_data}/{test_name}/block3D/HC"

# test_ad = f"/home/caizhihong/Data/{modalData}/Fold/{test_name}/AD"
# test_cn = f'/home/caizhihong/Data/{modalData}/Fold/{test_name}/HC'
# test_ad_block = f"/home/caizhihong/Data/{modalData}/Fold/{test_name}/block3D/AD"
# test_cn_block = f"/home/caizhihong/Data/{modalData}/Fold/{test_name}/block3D/HC"



# 输出文件路径
train_data_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/MutilBranch/{fold}/txt/{type_data}/train_data.txt"
train_label_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/MutilBranch/{fold}/txt/{type_data}/train_label.txt"
test_data_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/MutilBranch/{fold}/txt/{type_data}/test_data.txt"
test_label_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/MutilBranch/{fold}/txt/{type_data}/test_label.txt"

# train_data_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/valid_txt/valid_data.txt"
# train_label_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/valid_txt/valid_label.txt"
# test_data_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/valid_txt/test_data.txt"
# test_label_file = f"/home/caizhihong/CodeResult/{modalResult}/Fold/{fold}/valid_txt/test_label.txt"



# 用于存放训练和测试样本的列表
train_data = []
train_label = []
test_data = []
test_label = []

# 处理AD数据集
train_ad_samples = os.listdir(ad_dataset_dir)
train_ad_samples = sorted(train_ad_samples)

test_ad_samples = os.listdir(test_ad)
test_ad_samples = sorted(test_ad_samples)


test_data.extend([os.path.join(test_ad_block, "block%d", sample.replace(".nii", ".npy")) for sample in test_ad_samples])
test_label.extend([0] * len(test_ad_samples))  # AD的标签是0

# 其余样本添加到训练集
train_data.extend([os.path.join(ad_block_dir, "block%d", sample.replace(".nii", ".npy")) for sample in train_ad_samples])
train_label.extend([0] * len(train_ad_samples))

# 处理CN数据集
train_cn_samples = os.listdir(cn_dataset_dir)
train_cn_samples = sorted(train_cn_samples)

test_cn_samples = os.listdir(test_cn)
test_cn_samples = sorted(test_cn_samples)


test_data.extend([os.path.join(test_cn_block, "block%d", sample.replace(".nii", ".npy")) for sample in test_cn_samples])
test_label.extend([1] * len(test_cn_samples))  # CN的标签是1


train_data.extend([os.path.join(cn_block_dir, "block%d", sample.replace(".nii", ".npy")) for sample in train_cn_samples])
train_label.extend([1] * len(train_cn_samples))

# 将数据写入文件
with open(train_data_file, "w") as f:
    f.write("\n".join(train_data))
    # f.write("\n".join(train_data))

with open(train_label_file, "w") as f:
    f.write("\n".join(map(str, train_label)))

with open(test_data_file, "w") as f:
    f.write("\n".join(test_data))

with open(test_label_file, "w") as f:
    f.write("\n".join(map(str, test_label)))

print("数据集已分割并保存为文件。")