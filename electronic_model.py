import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(12, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 12)
        self.linear4 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o1 = self.relu(self.linear1(x))
        o2 = self.relu(self.linear2(o1))
        o3 = self.relu(self.linear3(o2))
        o4 = self.relu(self.linear4(o3))
        o5 = self.sigmoid(o4)
        return o5
