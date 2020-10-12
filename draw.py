
import numpy as np
import matplotlib.pyplot as plt



def main():
    fl_result_path = "C:\\Users\\yifanhu\\Documents\\ISPRS_RESULT.txt"
    center_result_path = "C:\\Users\\yifanhu\\Documents\\centralized_train.txt"
    fl_epoch, f_epoch = 1, 1
    fl_x, fl_y = [], []
    f_x, f_y = [], []
    with open(fl_result_path, "r") as fl:
        lines = fl.readlines()
        for line in lines:
            sp = line.split(" ")
            if sp[-2] == "test_accuracy:":
                # print(float(sp[-1]))
                fl_x.append(fl_epoch)
                fl_y.append(float(sp[-1]))
                fl_epoch += 1
    with open(center_result_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("test accuracy"):
                sp = line.split(" ")
                # print(float(sp[-1]))
                f_x.append(f_epoch)
                f_y.append(float(sp[-1]))
                f_epoch += 1

    plt.figure(figsize=(8, 4))
    plt.plot(fl_x, fl_y, color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("FedAvg Training Result")
    # plt.ylim(-1.2, 1.2)
    plt.show()


if __name__ == "__main__":
    main()