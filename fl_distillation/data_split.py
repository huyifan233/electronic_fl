import pandas as pd
from sklearn.preprocessing import StandardScaler



TRAIN_DATA_LEN = 54000
DATA_PATH_WIN_ROOT = "C:\\Users\\yifanhu\\Documents\\"
DATA_PATH_WIN = "C:\\Users\\yifanhu\\Documents\\smart_grid_stability_augmented.csv"

DATA_PATH = "/home/chunxin.hyf/smart_grid_stability_augmented.csv"
DATA_PATH_ROOT = "/home/chunxin.hyf/"

def data_split():
    data = pd.read_csv(DATA_PATH_WIN)
    map1 = {'unstable': 0, 'stable': 1}
    data['stabf'] = data['stabf'].replace(map1)
    data = data.sample(frac=1)
    test_data = data.iloc[TRAIN_DATA_LEN:, :]

    train_data_0 = data.iloc[:TRAIN_DATA_LEN//2, :]
    train_data_1 = data.iloc[TRAIN_DATA_LEN//2:TRAIN_DATA_LEN, :]

    test_data_df = pd.DataFrame(test_data)
    # test_data_df.to_csv(DATA_PATH_WIN_ROOT+"test_data.csv", index=0)
    # print(test_data_df)
    train_data_0_df = pd.DataFrame(train_data_0)
    train_data_1_df = pd.DataFrame(train_data_1)

    train_data_0_df.to_csv(DATA_PATH_WIN_ROOT+"train_data_0.csv", index=0)
    train_data_1_df.to_csv(DATA_PATH_WIN_ROOT+"train_data_1.csv", index=0)




def main():
    data_split()


if __name__ == "__main__":
    main()