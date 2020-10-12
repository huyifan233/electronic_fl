import torch
import torch.utils.data as data
import pandas as pd
import numpy as np


class ElectronicDataset(data.Dataset):
    def __init__(self, x, y):
        super(ElectronicDataset, self).__init__()
        self.x = np.array(x, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.len = self.x.shape[0]


    def __getitem__(self, item):
        item = item % self.len
        data, label = self.x[item], self.y[item]
        # data = torch.from_numpy(data)
        # label = torch.from_numpy(label)
        return data, label

    def __len__(self):
        return self.len