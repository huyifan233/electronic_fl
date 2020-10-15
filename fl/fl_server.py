import os
import time
import torch
import torch.nn as nn
from fl.fl_model import Net

GLOBAL_DIR = "./global_model_dir"
LOCAL_DIR = "./local_model_dir"
CLIENT_NUM = 2

if not os.path.exists(LOCAL_DIR):
    os.mkdir(LOCAL_DIR)

if not os.path.exists(GLOBAL_DIR):
    os.mkdir(GLOBAL_DIR)

# def weights_init(m):
#     if isinstance(m, (nn.Linear)):
#         nn.init.uniform_(m.weight)


def init_global_model(idx):
    init_global_model_path = os.path.join(GLOBAL_DIR, "global_model_{}".format(idx))
    if os.path.exists(init_global_model_path):
        return
    model = Net()
    # model.apply(weights_init)
    torch.save(model.state_dict(), init_global_model_path)

def aggregate(latest_model_paths, idx):

    other_model_pars = [torch.load(path) for path in latest_model_paths]
    model_pars = other_model_pars[0]
    for key in model_pars.keys():
        for i in range(1, len(latest_model_paths)):
            model_pars[key] += other_model_pars[i][key]
        model_pars[key] = torch.div(model_pars[key], len(latest_model_paths))
    # new_model.load_state_dict(model_pars)
    aggregated_model_path = os.path.join(GLOBAL_DIR, "global_model_{}".format(idx))
    torch.save(model_pars, aggregated_model_path)



def main():

    idx = 0
    init_global_model(idx)
    idx += 1
    print("Start Aggregating Server")

    while True:
        latest_model_paths = []
        for file in os.listdir(LOCAL_DIR):
            file_path = os.path.join(LOCAL_DIR,  file)
            if os.path.isdir(file_path):
                file_list = sorted(os.listdir(file_path), key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
                if len(file_list) == 0 or file_list[-1].split("_")[-1] != str("{}".format(idx-1)):
                    break
                latest_model_path = os.path.join(file_path, file_list[-1])
                latest_model_paths.append(latest_model_path)

        if len(latest_model_paths) == CLIENT_NUM:
            print("Execute {} aggregation...".format(idx))
            aggregate(latest_model_paths, idx)
            idx += 1
        else:
            time.sleep(0.1)




if __name__ == "__main__":
    main()