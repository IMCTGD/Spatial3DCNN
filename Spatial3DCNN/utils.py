import numpy as np
import nibabel as nib
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -*- coding: utf-8 -*-

# 切块
def cut_block(nii_filepath):
    # nii = nib.load('F:/swmADNI_941_S_1203_MR_MPR__GradWarp__B1_Correction_Br_20070801201750736_S25671_I63880.nii')
    nii = nib.load(nii_filepath)
    image = nii.get_fdata()
    print("image.shape:", image.shape)
    if image.shape[0] == 121 and image.shape[1] == 145 and image.shape[2] == 121:
        image_pad = np.pad(image, pad_width=((2, 2), (2, 3), (2, 2)), mode='constant', constant_values=(0, 0))
    elif image.shape[0] == 113 and image.shape[1] == 137 and image.shape[2] == 113:
        image_pad = np.pad(image, pad_width=((6, 6), (6, 7), (6, 6)), mode='constant', constant_values=(0, 0))
    print("image_pad.shape:", image_pad.shape)
    block_list = []
    for l in range(0, image_pad.shape[0], 25):
        for w in range(0, image_pad.shape[1], 25):
            if w == 10:
                print(50)
            for h in range(0, image_pad.shape[2], 25):
                temporary_block = image_pad[l:l + 25, w:w + 25, h:h + 25]
                block_list.append(temporary_block)
    return block_list


# 保存每个样本切好的块，save_path文件夹路径，filename_文件名，block_list块名
def save_block_list(save_path, filename, block_list):
    for i in range(len(block_list)):
        folder_path = os.path.join(save_path, 'block' + str(i))
        if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(folder_path)
        # filepath = save_path + str(i)
        filepath = os.path.join(folder_path, filename[:-4])
        np.save(filepath, block_list[i])


# 保存所有的文件,fileDir所要读取的nii文件夹，saveDir保存的文件夹
def save_all_nii_block(fileDir, saveDir):
    for file in os.listdir(fileDir):
        file_name = fileDir + "/" + file
        block_list = cut_block(file_name)

        save_block_list(save_path=saveDir, filename=file, block_list=block_list)


# fileDir1:种类1的block的文件路径
# fileDir2:种类2的block的文件路径
def get_data(fileDir1, fileDir2):
    data_List = []
    filepath_list = []
    labelLen_list = []
    subjectId_list = []
    filepath_list.append(fileDir1)  # block0到blockn的路径列表
    filepath_list.append(fileDir2)
    for fileDir in filepath_list:
        # 记录标签长度
        labelLen_list.append(len(os.listdir(fileDir)))
        # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        for file in os.listdir(fileDir):  # 读取block文件夹下面的文件格式为npy
            filepath = os.path.join(fileDir, file)  # 每个block文件下的数据路径
            subjectId_list.append(filepath[-14:-4])
            data = np.load(filepath)  # 读取数据
            data = np.expand_dims(data, axis=3)  # ?
            data_List.append(data)
    label_List = [0 for i in range(labelLen_list[0])]
    for i in range(labelLen_list[1]):
        label_List.append(1)
    return data_List, label_List

# MCIc对HC验证集
# def get_data(fileDir1, fileDir2):
#     data_List = []
#     filepath_list = []
#     labelLen_list = []
#     filepath_list.append(fileDir1)
#     filepath_list.append(fileDir2)
#     for fileDir in filepath_list:
#         # 记录标签长度
#         labelLen_list.append(39)
#         flag = 0
#         for file in os.listdir(fileDir):
#             flag = flag + 1
#             filepath = os.path.join(fileDir, file)
#             data = np.load(filepath)
#             data = np.expand_dims(data, axis=3)
#             data_List.append(data)
#             if flag == 39:
#                 break
#     label_List = [0 for i in range(39)]
#     for i in range(39):
#         label_List.append(1)
#     return data_List, label_List

# MCIc对MCInc验证集



def get_dataset(data_List, label_List, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((data_List, label_List))
    dataset = dataset.shuffle(10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat().prefetch(1)
    return dataset


if __name__ == "__main__":
    # 切块
    # filepath = '/home/caizhihong/T2ADNITrain/AD/wm002_S_0816.nii'
    # nii_file_path1 = '/home/sharedata/ADNI/MRI_509data/postprocessing/AD/swmADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction_Br_20070217010010760_S18402_I40732.nii'
    # block_list = cut_block(filepath)
    # print(block_list[0].shape)
    # block_list1 = cut_block(nii_file_path1)
    # print(block_list1[0].shape)

    # root_filepath = '/home/caizhihong/T2ADNITrain/AD'
    # save_Dir = '/home/caizhihong/T2ADNITrain/block_3D/AD'
    # save_all_nii_block(root_filepath, save_Dir)
    # root_filepath = '/home/caizhihong/T2ADNITrain/CN'
    # save_Dir = '/home/caizhihong/T2ADNITrain/block_3D/CN'
    # save_all_nii_block(root_filepath, save_Dir)
    # alldata_path_ad = f'/home/caizhihong/Data/ModalData/AllData/MRI/AD'
    # save_Dir = f'/home/caizhihong/Data/ModalData/AllData/MRI/block3D/AD'
    # save_all_nii_block(alldata_path_ad, save_Dir)
    # alldata_path_ad = f'/home/caizhihong/Data/ModalData/AllData/MRI/HC'
    # save_Dir = f'/home/caizhihong/Data/ModalData/AllData/MRI/block3D/HC'
    # save_all_nii_block(alldata_path_ad, save_Dir)

    # # PET数据切块
    modal = 'PETData'
    # fold_list = ['fold5_135', 'fold2_135', 'fold3_135', 'fold4_135']
    fold = 'fold1'
    test = 'test'
    # 切块训练集
    root_filepath = f'/home/caizhihong/Data/{modal}/Fold/{fold}/train/AD'
    save_Dir = f'/home/caizhihong/Data/{modal}/Fold/{fold}/block3D/train/AD'
    save_all_nii_block(root_filepath, save_Dir)
    root_filepath = f'/home/caizhihong/Data/{modal}/Fold/{fold}/train/HC'
    save_Dir = f'/home/caizhihong/Data/{modal}/Fold/{fold}/block3D/train/HC'
    save_all_nii_block(root_filepath, save_Dir)

    # valid cut block
    root_filepath = f'/home/caizhihong/Data/{modal}/Fold/{fold}/valid/AD'
    save_Dir = f'/home/caizhihong/Data/{modal}/Fold/{fold}/block3D/valid/AD'
    save_all_nii_block(root_filepath, save_Dir)
    root_filepath = f'/home/caizhihong/Data/{modal}/Fold/{fold}/valid/HC'
    save_Dir = f'/home/caizhihong/Data/{modal}/Fold/{fold}/block3D/valid/HC'
    save_all_nii_block(root_filepath, save_Dir)
    # #
    # # test cut block
    root_filepath = f'/home/caizhihong/Data/{modal}/Fold/{test}/AD'
    save_Dir = f'/home/caizhihong/Data/{modal}/Fold/{test}/block3D/AD'
    save_all_nii_block(root_filepath, save_Dir)
    #
    root_filepath = f'/home/caizhihong/Data/{modal}/Fold/{test}/HC'
    save_Dir = f'/home/caizhihong/Data/{modal}/Fold/{test}/block3D/HC'
    save_all_nii_block(root_filepath, save_Dir)

    # # 数据集测试
    # fileDir1 = 'F:/block_data/AD_Block/block0'
    # fileDir2 = 'F:/block_data/MClc_Block/block0'
    # data_list, label_list = get_data(fileDir1, fileDir2)
    # dataset = get_dataset(data_list, label_list, 4)
    # print(dataset)

