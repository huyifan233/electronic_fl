import os
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from electronic_dataset import ElectronicDataset
from electronic_model import Net
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold



CLIENT_ID = 0
GLOBAL_DIR = "./global_model_dir/"
LOCAL_DIR = "./local_model_dir/{}/".format(CLIENT_ID)
if not os.path.exists(LOCAL_DIR):
    os.mkdir(LOCAL_DIR)


DATA_PATH_WIN = "C:\\Users\\tchennech\\Documents\\electronic_fl\\smart_grid_stability_augmented.csv"
DATA_PATH = "/home/chunxin.hyf/smart_grid_stability_augmented.csv"
TRAIN_DATA_LEN = 54000
BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.0001

def load_dataset():
    data = pd.read_csv(DATA_PATH_WIN)
    map1 = {'unstable': 0, 'stable': '1'}
    data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)
    # data2 = data
    # f_most_correlated = data.corr().nlargest(14, 'stabf')['stabf'].index
    # print(data2.corr())


    X = data.iloc[:, :12]
    Y = data.iloc[:, 13]

    print(X.head())
    X_train = X.iloc[:TRAIN_DATA_LEN, :]
    Y_train = Y.iloc[:TRAIN_DATA_LEN]

    X_test = X.iloc[TRAIN_DATA_LEN:, :]
    Y_test = Y.iloc[TRAIN_DATA_LEN:]
    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train_dataset = ElectronicDataset(X_train, Y_train)
    # test_dataset = ElectronicDataset(X_test, Y_test)
    # return train_dataset, test_dataset
    X_train = torch.from_numpy(np.array(X_train, dtype=np.float32))
    Y_train = torch.from_numpy(np.array(Y_train, dtype=np.float32))
    X_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
    Y_test = torch.from_numpy(np.array(Y_test, dtype=np.float32))
    return X_train, Y_train, X_test, Y_test

def train(x_train, y_train, model, optimizer):

    model.train()
    for _ in range(EPOCH):
        pred = model(x_train)
        pred = pred.resize(pred.size()[0])
        loss = F.binary_cross_entropy(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(x_val, y_val, model):
    model.eval()
    pred_val = model(x_val)
    pred_val = pred_val.resize(pred_val.size()[0])
    loss_val = F.binary_cross_entropy(pred_val, y_val)
    pred_val[pred_val <= 0.5] = 0
    pred_val[pred_val > 0.5] = 1

    acc = torch.eq(pred_val, y_val).sum().float().item()
    acc /= y_val.size()[0]
    return acc, loss_val

def test(x_test, y_test, model):
    model.eval()
    pred_test = model(x_test)
    pred_test = pred_test.resize(pred_test.size()[0])
    loss_test = F.binary_cross_entropy(pred_test, y_test)
    pred_test[pred_test <= 0.5] = 0
    pred_test[pred_test > 0.5] = 1
    test_acc = torch.eq(pred_test, y_test).sum().float().item()
    return loss_test.item(), test_acc / pred_test.size()[0]

def load_global_model(epoch):

    global_model_path = GLOBAL_DIR + "global_model_{}".format(epoch)
    if os.path.exists(global_model_path):
        model = torch.load(global_model_path)
        return model
    return None


def save_local_model(epoch, model):

    save_local_model_path = LOCAL_DIR + "local_model_{}".format(epoch)
    torch.save(model, save_local_model_path)


def main():

    X_train, Y_train, X_test, Y_test = load_dataset()

    # model = Net()
    # optimizer = torch.optim.Adam(model.parameters())
    time.sleep(5)
    idx = 1
    out = 0
    while(out < 10):
        model = load_global_model(idx - 1)
        if model is not None:
            for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(X_train):
                optimizer = torch.optim.Adam(model.parameters())
                x_train, x_val = X_train[train_index], X_train[val_index]
                y_train, y_val = Y_train[train_index], Y_train[val_index]
                train(x_train, y_train, model, optimizer)
                acc, loss = validate(x_val, y_val, model)
                print("epoch: {}, val_loss: {}, val_accuracy: {}".format(idx, loss, acc))
                save_local_model(idx, model)
                idx += 1
            out += 1



    # loss, acc = test(X_test, Y_test, model)
    # print("test_loss: {}, test accuracy: {}".format(loss, acc))


if __name__ == "__main__":
    main()