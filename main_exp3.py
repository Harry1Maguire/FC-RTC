import pickle
import pcap_proc4
import pcap_proc6
from process import *
import numpy as np
from conmodel_coldstart import *
from conmodel_finetune import *
from supconloss import SupConLoss
from save_print import save_print_output

classific_mode = 'application'  # 分类模式 vpn/type/application
ipv = 4  # 数据集ip协议版本
target = 2  # 目标分类数目
localepoch = 1  # 客户端训练轮数
globalepoch = 100  # 全局训练轮数
mode = 'coldstart'  # 训练模式coldstrat/finetune
model_path = './exp2-finetunemodel/'  # 预训练模型路径
experiment_times = 11

if classific_mode == 'vpn':
    target = 2
    mod_str = 'vpn'
    fname_list = ['vpn', '20-vpn', '40-vpn', '60-vpn', '80-vpn']
elif classific_mode == 'type':
    target = 6
    mod_str = 'type'
    fname_list = ['traffic type', '20-type', '40-type', '60-type', '80-type']
elif classific_mode == 'application':
    target = 15
    mod_str = 'app'
    fname_list = ['application', '20-app', '40-app', '60-app', '80-app']
else:
    print('Inappropriate classification task!')
clientnum = target  # 客户端数目

# 存储用于分割数据集的hash字典的函数
def save_dict(dict1, dict2, dict3, hash_save_file):
    with open(hash_save_file, 'w') as f:
        f.write(str(dict1) + '\n')
        f.write(str(dict2) + '\n')
        f.write(str(dict3) + '\n')
    print('字典变量已保存到文件'+hash_save_file+'中。')

# 同上，读取字典的函数
def load_dict(filename):
    with open(filename, 'r') as f:
        dict1_str = f.readline()
        dict2_str = f.readline()
        dict3_str = f.readline()
    dict1 = eval(dict1_str)
    dict2 = eval(dict2_str)
    dict3 = eval(dict3_str)
    return dict1, dict2, dict3

def dataset_train(obj, data_dict):
    train_dict = {'shared':[], 'val':[], 'test':[]}
    for key in data_dict:
        list_all = data_dict[key]
        HDRs = list_all[0]
        PAYs = list_all[1]
        tokens = list_all[2]
        div1 = int(len(HDRs) * 0.73)
        div2 = int(len(HDRs) * 0.75)
        div3 = int(len(HDRs) * 0.85)
        train_dict[key] = [HDRs[:div1], PAYs[:div1], tokens[:div1]]
        if train_dict['shared'] == []:
            train_dict['shared'] = [HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2]]
        else:
            for i in range(len(train_dict['shared'])):
                a = [HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2]]
                train_dict['shared'][i] = np.vstack((train_dict['shared'][i], a[i]))
        if train_dict['val'] == []:
            train_dict['val'] = [HDRs[div2:div3], PAYs[div2:div3], tokens[div2:div3]]
        else:
            for i in range(len(train_dict['val'])):
                a = [HDRs[div2:div3], PAYs[div2:div3], tokens[div2:div3]]
                train_dict['val'][i] = np.vstack((train_dict['val'][i], a[i]))
        if train_dict['test'] == []:
            train_dict['test'] = [HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2]]
        else:
            for i in range(len(train_dict['test'])):
                a = [HDRs[div3:], PAYs[div3:], tokens[div3:]]
                train_dict['test'][i] = np.vstack((train_dict['test'][i], a[i]))
    for j in range(len(train_dict[obj])):
        train_dict[obj][j] = np.vstack((train_dict['shared'][j], train_dict[obj][j]))
    HDR_obj = train_dict[obj][0]
    PAY_obj = train_dict[obj][1]
    token_obj = train_dict[obj][2].squeeze()
    train_set = myPreDataSet(HDR_obj, PAY_obj, token_obj)
    HDR_val = train_dict['val'][0]
    PAY_val = train_dict['val'][1]
    token_val = train_dict['val'][2].squeeze()
    val_set = myPreDataSet(HDR_val, PAY_val, token_val)
    HDR_test = train_dict['test'][0]
    PAY_test = train_dict['test'][1]
    token_test = train_dict['test'][2].squeeze()
    test_set = myPreDataSet(HDR_test, PAY_test, token_test)
    return train_set, val_set, test_set




def fedTask_unblanced(filename_list, tarnum, cnum, mode, exp_times):
    train_list = {}
    val_list = {}
    test_list = {}
    datasets_unbalanced = {}
    exp_times = str(exp_times)
    hash_file = 'hash-exp2-' + exp_times
    if not os.path.exists(f'{hash_file}.txt'):
        for filename in filename_list:
            if filename == filename_list[0]:  # 完整流量数据文件
                with open(f"{filename}.pkl", "rb") as f:
                    [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    if obj not in datasets_unbalanced:
                        datasets_unbalanced[obj] = []
                    list1, list2, list3 = creat_hash_for_datasplit(streams[obj])
                    datasets_unbalanced[obj].append(list(stream_to_data_unbalanced_type(streams[obj], fdict[obj], list1,list2, list3)))
                    train_list.setdefault(obj, list1)
                    val_list.setdefault(obj, list2)
                    test_list.setdefault(obj, list3)
                hash_save_file = 'hash-exp3-' + exp_times + '.txt'
                save_dict(train_list, val_list, test_list, hash_save_file)
            else:
                with open(f"{filename}.pkl", "rb") as f:
                    [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    if obj not in datasets_unbalanced:
                        datasets_unbalanced[obj] = []
                    list1 = train_list[obj]
                    list2 = val_list[obj]
                    list3 = test_list[obj]
                    datasets_unbalanced[obj].append(list(stream_to_data_unbalanced_type(streams[obj], fdict[obj], list1, list2, list3)))
    else:
        for filename in filename_list:
            with open(f"{filename}.pkl", "rb") as f:
                [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    dict1, dict2, dict3 = load_dict(hash_file + '.txt')
                    list1 = dict1[obj]
                    list2 = dict2[obj]
                    list3 = dict3[obj]
                    if obj not in datasets_unbalanced:
                        datasets_unbalanced[obj] = list(stream_to_data_unbalanced_type(streams[obj], fdict[obj], list1, list2, list3))
                    else:
                        for j in range(len(datasets_unbalanced[obj])):
                            a = list(stream_to_data_unbalanced_type(streams[obj], fdict[obj], list1, list2, list3))
                            datasets_unbalanced[obj][j] = np.vstack((datasets_unbalanced[obj][j], a[j]))
    category_count = len(datasets_unbalanced)
    list_lengths = {key: len(value) for key, value in datasets_unbalanced.items() if isinstance(value, list)}
    category_names = list(datasets_unbalanced.keys())
    Global_Model = myDISTILLER_coldstart(tarnum).to(device)
    Clients_coldstart = []
    for i in range(cnum):
        dataset = dataset_train(category_names[i], datasets_unbalanced)
        Clients_coldstart.append(Client_coldstart(
            i,
            [dataset[a] for a in range(3)],
            tarnum,
        ))
    for e in range(globalepoch):
        global_dict = Global_Model.state_dict()
        for client in Clients_coldstart:
            client.model.load_state_dict(global_dict)
        cLoss = [[] for _ in range(4)]
        for _, gparam in Global_Model.named_parameters():
            gparam.data.zero_()
        for client in Clients_coldstart:
            tmpLoss = client.train(localepoch)
            for j in range(4):
                cLoss[j].append(tmpLoss[j])
            for gParam, cParam in zip(Global_Model.parameters(), client.model.parameters()):
                gParam.data += cParam.data / cnum
        cLoss = [np.mean(cLoss[i]) for i in range(4)]
        print(cLoss)
    torch.save(Global_Model.state_dict(), './model/coldstart_exp3_without' + classific_mode +'_' + exp_times +'.pth')
    tloss = []
    tacc = []
    for client in Clients_coldstart:
        tmp = client.test()
        tloss.append(tmp[0])
        tacc.append(tmp[1])
    tloss = np.mean(tloss)
    tacc = np.mean(tacc)
    print([tloss, tacc])

def DatasetGenerate(filename, ipversion=4):
    fpath = f'.\\{filename}\\'
    files = os.listdir(fpath)
    fdict = {files[i]: i for i in range(len(files))}
    print('ok')
    streams = {}
    for i in range(len(files)):
        obj = files[i]
        streams[obj] = dict()
        objfiles = os.listdir(os.path.join(fpath, obj))
        for file in objfiles:
            # print(os.path.join(obj, file))
            if ipversion == 6:
                streams[obj].update(pcap_proc6.pcap_proc(os.path.join(fpath, obj, file)))
            else:
                streams[obj].update(pcap_proc4.pcap_proc(os.path.join(fpath, obj, file)))
        print(len(streams[obj]))
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump([streams, files, fdict], f)

if __name__ == "__main__":
    for fname in fname_list:
        if not os.path.exists(f'{fname}.pkl'):
            DatasetGenerate(fname, ipv)
    for i in range(1, experiment_times):
        fedTask_unblanced(fname_list, target, clientnum, mode=mode, exp_times=i)
'''
    dataset = datasets[0].copy()
    for i in range(1, len(datasets)):
        dataset[0] = dataset[0] + datasets[i][0]
        dataset[1] = dataset[1] + datasets[i][1]
        dataset[2] = dataset[2] + datasets[i][2]
        dataset[3] = dataset[3] + datasets[i][3]
'''