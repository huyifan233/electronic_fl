import os
import pandas as pd
import torch
import time
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from fl.fl_model import Net


CLIENT_ID = 1
GLOBAL_DIR = os.path.join("./global_model_dir", str(CLIENT_ID))
LOCAL_DIR = os.path.join("./local_model_dir", str(CLIENT_ID))
if not os.path.exists(LOCAL_DIR):
    os.mkdir(LOCAL_DIR)


DATA_PATH_WIN = "C:\\Users\\tchennech\\Documents\\electronic_fl\\train_data_1.csv"
TEST_DATA_PATH_WIN = "C:\\Users\\tchennech\\Documents\\electronic_fl\\test_data.csv"
DATA_PATH = "/home/chunxin.hyf/train_data_1.csv"
TRAIN_DATA_LEN = 27000
TEST_DATA_LEN = 6000
BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.0001

def load_dataset():
    data = pd.read_csv(DATA_PATH_WIN)
    test_data = pd.read_csv(TEST_DATA_PATH_WIN)
    # map1 = {'unstable': 0, 'stable': 1}
    # data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)
    test_data = test_data.sample(frac=1)

    X = data.iloc[:, :12]
    Y = data.iloc[:, 13]
    test_X = test_data.iloc[:, :12]
    test_Y = test_data.iloc[:, 13]

    X_train = X.iloc[:TRAIN_DATA_LEN, :]
    Y_train = Y.iloc[:TRAIN_DATA_LEN]
    X_test = test_X.iloc[:TEST_DATA_LEN, :]
    Y_test = test_Y.iloc[:TEST_DATA_LEN]


    X_train = X_train.values
    Y_train = Y_train.values
    X_test = X_test.values
    Y_test = Y_test.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.from_numpy(np.array(X_train, dtype=np.float32))
    Y_train = torch.from_numpy(np.array(Y_train, dtype=np.float32))
    X_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
    Y_test = torch.from_numpy(np.array(Y_test, dtype=np.float32))
    # return X_train, Y_train, X_test, Y_test
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

    global_model_path = os.path.join(GLOBAL_DIR, "global_model_{}".format(epoch))
    if os.path.exists(global_model_path):
        print("Load global model: {}".format(global_model_path))
        time.sleep(0.5)
        model_pars = torch.load(global_model_path)
        model = Net()
        model.load_state_dict(model_pars)
        return model
    return None


def save_local_model_pars(epoch, model_pars):
    save_local_model_path = os.path.join(LOCAL_DIR, "local_model_{}".format(epoch))
    print("Save local model: {}".format(save_local_model_path))
    torch.save(model_pars, save_local_model_path)


def main():

    X_train, Y_train, X_test, Y_test = load_dataset()

    # model = Net()
    # optimizer = torch.optim.Adam(model.parameters())


    out = 0
    while(out < 10):
        model = load_global_model(out)
        if model is not None:
            optimizer = torch.optim.Adam(model.parameters())
            for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(X_train):
                x_train, x_val = X_train[train_index], X_train[val_index]
                y_train, y_val = Y_train[train_index], Y_train[val_index]
                train(x_train, y_train, model, optimizer)
                acc, loss = validate(x_val, y_val, model)
            print("epoch: {}, val_loss: {}, val_accuracy: {}".format(out, loss, acc))
            save_local_model_pars(out, model.state_dict())
            out += 1



    loss, acc = test(X_test, Y_test, model)
    print("test_loss: {}, test accuracy: {}".format(loss, acc))


if __name__ == "__main__":
    main()