import os
import yaml

def cartesian(*para):
    """
    生成给定参数的笛卡尔积
    返回一个二维列表
    """
    combinations = [[]]
    for item in range(len(para)):
        sub_c = []
        for s in combinations:
            for i in para[item]:
                sub_c.append(s + [i])
        combinations = sub_c
    return combinations

def save_configs(para, path):
    if not os.path.exists(path):
        os.makedirs(path)
    config = {'data_dir':para[0], 'batch_size':para[1], 'epochs':para[2],'lr':para[3],'weight_decay':para[4], 'pretrained': para[5], 'fine_tune': para[6]}
    with open(os.path.join(path, 'alexnet.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
        

# data_dir = ["./data/caltech101"]
# batch_size = [16, 32, 64]
# epochs = [30]
# lr = [0.00005, 0.0001, 0.00015, 0.0002, 0.0003]
# weight_decay = [0.0005]
# pretrained = [True, False]
# fine_tune = [True]

# combination = cartesian(data_dir,batch_size,epochs,lr,weight_decay,pretrained,fine_tune)

combination = [['./data/caltech101', 32, 50, 0.0003, 0.0005, True, True], ['./data/caltech101', 16, 50, 0.0001, 0.0005, True, True], ['./data/caltech101', 16, 50, 0.0002, 0.0005, True, True]]

for i, para in enumerate(combination):
    save_configs(para, f'model_data/{i+1+30}')