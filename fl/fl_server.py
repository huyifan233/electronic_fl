import os
import time
import torch
from fl.fl_model import Net

GLOBAL_DIR = "./global_model_dir/"
LOCAL_DIR = "./local_model_dir/"

def init_global_model(idx):
    init_global_model_path = GLOBAL_DIR + "global_model_{}".format(idx)
    model = Net()
    torch.save(model, init_global_model_path)

def aggregate(latest_model_paths, idx):
    model = torch.load(latest_model_paths[0])
    for key in model.keys():
        for i in range(1, len(latest_model_paths)):
            other_model = torch.load(latest_model_paths[i])
            model[key] += other_model[i][key]
        model[key] = torch.div(model[key], len(latest_model_paths))
    aggregated_model_path = GLOBAL_DIR + "global_model_{}".format(idx)
    torch.save(model, aggregated_model_path)



def main():

    idx = 0
    init_global_model(idx)
    idx += 1
    print("Start Aggregating Server")
    while True:
        latest_model_paths = []
        is_ok = True
        if len(os.listdir(LOCAL_DIR)) == 0:
            is_ok = False
            continue
        for file in os.listdir(LOCAL_DIR):
            if os.path.isdir(file):
                file_list = sorted(os.listdir(file))
                if len(file_list) == 0:
                    is_ok = False
                    break
                if len(latest_model_paths) == 0:
                    latest_model_paths.append(file_list[-1])
                elif latest_model_paths[-1] != file_list[-1]:
                    is_ok = False
                    break

        if is_ok:
            aggregate(latest_model_paths, idx)
            idx += 1
        else:
            time.sleep(0.1)




if __name__ == "__main__":
    main()