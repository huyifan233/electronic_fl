import os
import pandas as pd
from sklearn.preprocessing import StandardScaler



TRAIN_DATA_LEN = 54000
DATA_PATH_WIN_ROOT = "C:\\Users\\tchennech\\Documents\\electronic_fl"
DATA_PATH_WIN = "C:\\Users\\tchennech\\Documents\\electronic_fl\\smart_grid_stability_augmented.csv"

DATA_PATH = "/home/chunxin.hyf/smart_grid_stability_augmented.csv"
DATA_PATH_ROOT = "/home/chunxin.hyf/"
# 0: 38280, 1: 21720
FIX_LEN = 27000
def data_split():
    data = pd.read_csv(DATA_PATH_WIN)
    map1 = {'unstable': 0, 'stable': 1}
    data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)
    # test_data = data.iloc[TRAIN_DATA_LEN:, :]
    data_0 = data[data['stabf'].isin([0])]
    data_1 = data[data['stabf'].isin([1])]
    # train_data_0 = data.iloc[:TRAIN_DATA_LEN//2, :]
    # train_data_1 = data.iloc[TRAIN_DATA_LEN//2:TRAIN_DATA_LEN, :]
    num_1 = int(FIX_LEN*0.9)
    # print(len(data_0))
    # print(len(data_1))
    train_data_0_noniid = pd.concat([pd.DataFrame(data_0.iloc[:num_1, :]), pd.DataFrame(data_1.iloc[:FIX_LEN-num_1, :])], ignore_index=True, axis=0)
    train_data_0_noniid = train_data_0_noniid.sample(frac=1)
    print(len(train_data_0_noniid))
    num_2 = int(FIX_LEN*0.2)
    train_data_1_noniid = pd.concat([pd.DataFrame(data_0.iloc[:num_2, :]), pd.DataFrame(data_1.iloc[:FIX_LEN - num_2, :])],
                             ignore_index=True, axis=0)
    train_data_1_noniid = train_data_1_noniid.sample(frac=1)
    print(len(train_data_1_noniid))
    # test_data_df = pd.DataFrame(test_data)
    # test_data_df.to_csv(DATA_PATH_WIN_ROOT+"test_data.csv", index=0)
    # print(test_data_df)
    # train_data_0_noniid_df = pd.DataFrame(train_data_0_noniid)
    # train_data_1_noniid_df = pd.DataFrame(train_data_1_noniid)
    #
    train_data_0_noniid.to_csv(os.path.join(DATA_PATH_WIN_ROOT,"train_data_0_noniid.csv"), index=0)
    train_data_1_noniid.to_csv(os.path.join(DATA_PATH_WIN_ROOT,"train_data_1_noniid.csv"), index=0)
    # test_data_df.to_csv(DATA_PATH_WIN_ROOT+"test_data.csv", index=0)



def main():
    data_split()


if __name__ == "__main__":
    main()