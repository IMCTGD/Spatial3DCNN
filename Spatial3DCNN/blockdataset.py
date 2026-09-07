from tensorflow.keras.callbacks import Callback
import numpy as np
import numpy.random as random
import random


# 翻转
class randomflip(object):
    def __call__(self, data):
        # print(data.shape)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=0)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=1)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=2)
        return data


# 旋转180度
class randomflip180(object):
    def __call__(self, data):
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=1)
            data = np.flip(data, axis=2)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=2)
            data = np.flip(data, axis=0)
        return data


# 噪声加0~0.1
class noisy(object):
    def __init__(self, radio):
        self.radio = radio

    def __call__(self, data):
        l, w, h, _ = data.shape
        num = int(l * w * h * self.radio)
        for _ in range(num):
            x = np.random.randint(0, l)
            y = np.random.randint(0, w)
            z = np.random.randint(0, h)
            data[x, y, z, 0] = data[x, y, z, 0] + np.random.uniform(0, 0.05)
        return data


class blockdataset(object):
    def __init__(self, data, label, transform, status, batch_size, shuffle):
        self.data = data
        self.label = label
        self.status = status
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return (self.data[item], self.label[item])

    def data_generator(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        datalen = self.__len__()
        while 1:
            if self.shuffle:
                shuffle = np.random.permutation(self.__len__())
                data = self.data[shuffle]
                label = self.label[shuffle]
            else:
                data = self.data
                label = self.label
            for i in range(0, datalen, batch_size):
                x = data[i:min(datalen, i + batch_size)]
                y = label[i:min(datalen, i + batch_size)]
                if self.status == 'train':
                    for j in range(len(x)):
                        for f in self.transform:
                            x[j] = f(x[j])
                yield (x, y)


if __name__ == '__main__':
    data = np.ones((10, 5))
    label = np.ones((10, 1))
    dataset = blockdataset(data, label, transform=[], status='train', batch_size=2, shuffle=True)
    data_gen = dataset.data_generator()
    for i in range(10):
        a = next(data_gen)
        print(a)
