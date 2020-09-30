'''
Date: 2020-09-28 14:41:57
LastEditors: Tianling Lyu
LastEditTime: 2020-09-29 14:46:01
FilePath: \gesture_classification\dataset.py
'''

import random
import pickle

from mxnet.gluon.data import dataset
import numpy as np

class DVPickleDataset(dataset.Dataset):
    def __init__(self, filepath, n_frame, n_label=11, shuffle=True):
        with open(filepath, "rb") as f:
            self.__data = pickle.load(f)
        self.n_frame = n_frame
        self.n_label = n_label
        self.__items = []
        for i in range(len(self.__data)):
            images = self.__data[i]
            for key, value in images.items():
                nimg = len(value)
                for j in range(nimg-n_frame):
                    self.__items.append([i, key, j])
        if shuffle:
            random.shuffle(self.__items)
    
    def __getitem__(self, idx):
        item = self.__items[idx]
        label = np.eye(self.n_label)[int(item[1]) - 1]
        img = []
        for i in range(self.n_frame):
            img.append(np.expand_dims(self.__data[item[0]][item[1]][item[2]+i], 0))
        img = np.concatenate(img, axis=0)
        return img.astype(np.float32), label.astype(np.float32)
    
    def __len__(self):
        return len(self.__items)
