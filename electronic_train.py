import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from electronic_dataset import ElectronicDataset
from electronic_model import Net
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

DATA_PATH_WIN = "C:\\Users\\yifanhu\\Documents\\smart_grid_stability_augmented.csv"
DATA_PATH = "/home/chunxin.hyf/smart_grid_stability_augmented.csv"
TRAIN_DATA_LEN = 54000
BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.0001

def consturct_dataset():
    data = pd.read_csv(DATA_PATH)
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


def main():

    # train_dataset, test_dataset = consturct_dataset()
    X_train, Y_train, X_test, Y_test = consturct_dataset()
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


    model = Net()
    optimizer = torch.optim.Adam(model.parameters())
    # for idx, (batch_data, batch_label) in enumerate(train_dataloader):
    #     for _ in range(EPOCH):
    #         pred = model(batch_data)
    #         loss = F.binary_cross_entropy(pred, batch_label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print("idx:{}, loss: {}".format(idx, loss.item()))
    idx = 1
    for _ in range(5):
        for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(X_train):
            x_train, x_val = X_train[train_index], X_train[val_index]
            y_train, y_val = Y_train[train_index], Y_train[val_index]
            model.train()
            for _ in range(EPOCH):
                pred = model(x_train)
                pred = pred.resize(pred.size()[0])
                loss = F.binary_cross_entropy(pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            pred_val = model(x_val)
            pred_val = pred_val.resize(pred_val.size()[0])
            loss_val = F.binary_cross_entropy(pred_val, y_val)
            print("epoch: {}, loss: {}".format(idx, loss_val))
            pred_val[pred_val <= 0.5] = 0
            pred_val[pred_val > 0.5] = 1

            acc = torch.eq(pred_val, y_val).sum().float().item()
            acc /= y_val.size()[0]
            print(acc)
            idx += 1
    model.eval()
    pred_test = model(X_test)
    pred_test = pred_test.resize(pred_test.size()[0])
    loss_test = F.binary_cross_entropy(pred_test, Y_test)
    print("loss_test: {}".format(loss_test.item()))
    pred_test[pred_test <= 0.5] = 0
    pred_test[pred_test > 0.5] = 1
    test_acc = torch.eq(pred_test, Y_test).sum().float().item()
    print("test accuracy: {}".format(test_acc / pred_test.size()[0]))

    #     val_pred = model(x_val)
    #     loss_fn = torch.nn.BCELoss()
    #     loss_val = loss_fn(val_pred, y_val)
    #     print("loss_val: ", loss_val.item())
        # classifier.fit(x_train, y_train, epochs=50, verbose=0)
        # classifier_loss, classifier_accuracy = classifier.evaluate(x_val, y_val)
        # print(f'Round {cross_val_round} - Loss: {classifier_loss:.4f} | Accuracy: {classifier_accuracy * 100:.2f} %')
        # cross_val_round += 1
    # model.eval()
    # acc = 0
    # for idx, (test_batch_data, test_batch_label) in enumerate(test_dataloader):
    #     pred = model(test_batch_data)
    #     # print(pred, batch_label)
    #     pred = pred.resize(test_batch_label.size()[0])
    #     for idx2 in range(pred.size()[0]):
    #         # print(pred[idx][0])
    #         pred[idx2] = 0 if pred[idx2] <= 0.5 else 1
    #     # print(pred, test_batch_label)
    #     # print(pred.size(), test_batch_label.size())
    #     # print(torch.eq(pred, test_batch_label))
    #     acc += torch.eq(pred, test_batch_label).sum().float().item()
    #
    # print("accuracy: ", acc / len(test_dataloader))

if __name__ == "__main__":
    main()