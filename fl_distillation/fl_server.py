import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from fl.fl_model import Net
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler

GLOBAL_DIR = "./global_model_dir"
LOCAL_DIR = "./local_model_dir"
if not os.path.exists(LOCAL_DIR):
    os.mkdir(LOCAL_DIR)

if not os.path.exists(GLOBAL_DIR):
    os.mkdir(GLOBAL_DIR)
CLIENT_NUM = 2
EPOCH = 50
TRAIN_DATA_LEN = 27000
DATA_PATH_ROOT_WIN = "C:\\Users\\tchennech\\Documents\\electronic_fl\\"

if not os.path.exists(LOCAL_DIR):
    os.mkdir(LOCAL_DIR)

if not os.path.exists(GLOBAL_DIR):
    os.mkdir(GLOBAL_DIR)
    for client_id in range(CLIENT_NUM):
        path = GLOBAL_DIR + str(client_id)
        os.mkdir(path)

def validate(x_val, y_val, model):

    pred_val = model(x_val)
    pred_val = pred_val.resize(pred_val.size()[0])
    loss_val = F.binary_cross_entropy(pred_val, y_val)
    pred_val[pred_val <= 0.5] = 0
    pred_val[pred_val > 0.5] = 1

    acc = torch.eq(pred_val, y_val).sum().float().item()
    acc /= y_val.size()[0]
    return acc, loss_val

def load_dataset(client_id):
    DATA_PATH_WIN = DATA_PATH_ROOT_WIN + "train_data_{}.csv".format(client_id)
    data = pd.read_csv(DATA_PATH_WIN)
    # map1 = {'unstable': 0, 'stable': 1}
    # data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)

    X = data.iloc[:, :12]
    Y = data.iloc[:, 13]


    X_train = X.iloc[:TRAIN_DATA_LEN, :]
    Y_train = Y.iloc[:TRAIN_DATA_LEN]

    X_train = X_train.values
    Y_train = Y_train.values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    X_train = torch.from_numpy(np.array(X_train, dtype=np.float32))
    Y_train = torch.from_numpy(np.array(Y_train, dtype=np.float32))

    return X_train, Y_train

def init_global_model(idx):
    model = Net()
    for client_id in range(CLIENT_NUM):
        init_global_model_path = os.path.join(GLOBAL_DIR, str(client_id))
        if not os.path.exists(init_global_model_path):
            os.mkdir(init_global_model_path)
        torch.save(model.state_dict(), os.path.join(init_global_model_path, "global_model_{}".format(idx)))

def train(x_train, y_train, model, optimizer, other_model_list):


    model.train()
    for _ in range(EPOCH):
        pred = model(x_train)
        pred = pred.resize(pred.size()[0])
        distillation_loss = 0
        for other_model in other_model_list:
            other_model.train()
            other_model_pred = other_model(x_train)
            other_model_pred = other_model_pred.resize(other_model_pred.size()[0])
            distillation_loss += F.kl_div(pred, other_model_pred, reduction='batchmean')

        loss = F.binary_cross_entropy(pred, y_train) + distillation_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def distillation(latest_model_paths, idx):


    for client_id in range(CLIENT_NUM):
        X_train, Y_train = load_dataset(client_id)
        main_model_pars = None
        other_model_list = []
        for path in latest_model_paths:
            print(latest_model_paths)
            if path.split("\\")[-2] == str(client_id):
                main_model_pars = torch.load(path)
            else:
                 for path in latest_model_paths:
                    other_model = Net()
                    other_model.load_state_dict(torch.load(path))
                    other_model_list.append(other_model)

        model = Net()
        model.load_state_dict(main_model_pars)
        optimizer = torch.optim.Adam(model.parameters())
        for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(X_train):
            x_train, x_val = X_train[train_index], X_train[val_index]
            y_train, y_val = Y_train[train_index], Y_train[val_index]
            train(x_train, y_train, model, optimizer, other_model_list)
            acc, loss = validate(x_val, y_val, model)
            print("distillation_val_loss: {}, distillation_val_accuracy: {}".format(loss, acc))

        # new_model.load_state_dict(model_pars)
        aggregated_model_path = os.path.join(os.path.join(GLOBAL_DIR, str(client_id)), "global_model_{}".format(idx))
        torch.save(model.state_dict(), aggregated_model_path)



def main():

    idx = 0
    init_global_model(idx)
    idx += 1
    print("Start Aggregating Server")

    while True:
        latest_model_paths = []
        for file in os.listdir(LOCAL_DIR):
            file_path = os.path.join(LOCAL_DIR, file)
            if os.path.isdir(file_path):
                file_list = sorted(os.listdir(file_path), key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
                if len(file_list) == 0 or file_list[-1].split("_")[-1] != str("{}".format(idx-1)):
                    break
                latest_model_path = os.path.join(file_path, file_list[-1])
                latest_model_paths.append(latest_model_path)

        if len(latest_model_paths) == CLIENT_NUM:
            print("Execute {} aggregation...".format(idx))

            distillation(latest_model_paths, idx)
            idx += 1
        else:
            time.sleep(0.1)




if __name__ == "__main__":
    main()