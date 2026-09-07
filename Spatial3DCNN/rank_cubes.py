import csv
import os

def sort_data(dataset):
    fold = 'fold1'
    result = 'PETResult'
    acc_sort_path = f'/home/caizhihong/CodeResult/{result}/Fold/{fold}/log/ADvsHC/valid.csv'
    # acc_sort_path = '/home/caizhihong/lgy_data/p-score/czh_log/fold1/ADvsHC/valid_new.csv'

    with open(acc_sort_path, 'r') as csvfile:
        data_list = []
        reader = csv.reader(csvfile)
        # reader对象本身不是一个列表或字符串，而是一个迭代器。要打印出文件中的数据，您可以遍历reader对象并逐行打印，例如：
        for row in reader:
            print(row)
            data_list.append(row)
        # print("**********************切片前*******************")
        # print(data_list)
        # 去掉列名 'block', 'val_acc', 'val_loss', 'test_acc', 'test_loss'
        data_list = data_list[1:]
        # print("**********************切片后*******************")
        # print(data_list)
        data_list = sorted(data_list, key=lambda x: float(x[1]), reverse=True)
        print("**********************排序后****************")
        print(data_list)
    with open(f'/home/caizhihong/CodeResult/{result}/Fold/{fold}/log/ADvsHC/sort.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['block', 'val_acc', 'val_loss', 'test_acc', 'test_loss'])
        writer.writerows(data_list)




if __name__ == '__main__':
    dataset = 'oasis'
    sort_data(dataset)
