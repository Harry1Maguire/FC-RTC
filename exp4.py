import pickle
import pcap_proc4
import pcap_proc6
from process import *
from conmodel_coldstart import *
from conmodel_finetune import *
from supconloss import SupConLoss
from save_print import save_print_output

classific_mode = 'type'  # 分类模式 vpn/type/application
ipv = 4  # 数据集ip协议版本
target = 6  # 目标分类数目
clientnum = 4  # 客户端数目
localepoch = 1  # 客户端训练轮数
globalepoch = 100  # 全局训练轮数
mode = 'coldstart'  # 训练模式coldstrat/finetune
model_path = './exp2-finetunemodel/'  # 预训练模型路径
experiment_times = 10

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


def fedTask_mix(filename_list, tarnum, cnum, mode, exp_times):
    train_list = {}
    val_list = {}
    test_list = {}
    datasets = []
    unblanced_data = {}
    exp_times = str(exp_times)
    hash_file = 'hash-exp2-' + exp_times
    if not os.path.exists(f'{hash_file}.txt'):
        for filename in filename_list:
            if filename == filename_list[0]:  # 完整流量数据文件
                with open(f"{filename}.pkl", "rb") as f:
                    [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    list1, list2, list3 = creat_hash_for_datasplit(streams[obj])
                    datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
                    train_list.setdefault(obj, list1)
                    val_list.setdefault(obj, list2)
                    test_list.setdefault(obj, list3)
                hash_save_file = 'hash-exp2-' + exp_times + '.txt'
                save_dict(train_list, val_list, test_list, hash_save_file)
            else:
                with open(f"{filename}.pkl", "rb") as f:
                    [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    list1 = train_list[obj]
                    list2 = val_list[obj]
                    list3 = test_list[obj]
                    datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
    else:
        for filename in filename_list:
            unblanced_data[filename] = []
            if filename == filename_list[0]:  # 完整流量数据文件
                with open(f"{filename}.pkl", "rb") as f:
                    [streams, files, fdict] = pickle.load(f)
                    for ij in range(4):
                        datasets = []
                        unblanced_data[filename + str(ij)] = []
                        for obj in files:
                            dict1, dict2, dict3 = load_dict(hash_file + '.txt')
                            list1 = dict1[obj]
                            list2 = dict2[obj]
                            list3 = dict3[obj]
                            datasets.append(list(stream_to_data_share(streams[obj], fdict[obj], list1, list2, list3, ij)))
                        dataset = datasets[0].copy()
                        for k in range(1, len(datasets)):
                            dataset[0] = dataset[0] + datasets[k][0]
                            dataset[1] = dataset[1] + datasets[k][1]
                            dataset[2] = dataset[2] + datasets[k][2]
                        unblanced_data[filename + str(ij)] = dataset
            else:
                with open(f"{filename}.pkl", "rb") as f:
                    datasets = []
                    [streams, files, fdict] = pickle.load(f)
                    for obj in files:
                        dict1, dict2, dict3 = load_dict(hash_file + '.txt')
                        list1 = dict1[obj]
                        list2 = dict2[obj]
                        list3 = dict3[obj]
                        datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
                    dataset = datasets[0].copy()
                    for k in range(1, len(datasets)):
                        dataset[0] = dataset[0] + datasets[k][0]
                        dataset[1] = dataset[1] + datasets[k][1]
                        dataset[2] = dataset[2] + datasets[k][2]
                    unblanced_data[filename] = dataset

    if mode == 'coldstart':
        Global_Model = myDISTILLER_coldstart(tarnum).to(device)
        Clients_coldstart = []
        for i in range(cnum):
            train_set = unblanced_data[filename_list[i+1]] + unblanced_data[filename_list[0] + str(i)]
            print(train_set)
            Clients_coldstart.append(Client_coldstart(
                i,
                [train_set[j].part(1. / clientnum) for j in range(3)],
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
        torch.save(Global_Model.state_dict(), './model/normal_' + classific_mode + '_' + exp_times +'.pth')
        tloss = []
        tacc = []
        for client in Clients_coldstart:
            tmp = client.test()
            tloss.append(tmp[0])
            tacc.append(tmp[1])
        tloss = np.mean(tloss)
        tacc = np.mean(tacc)
        print([tloss, tacc])

    else:
        Global_Model = myDISTILLER_finetune(tarnum).to(device)
        Clients_finetune = []
        for i in range(cnum):
            Clients_finetune.append(Client_finetune(
                i,
                [dataset[i].part(1. / clientnum) for i in range(3)],
                tarnum,
            ))
        for e in range(globalepoch):
            global_dict = Global_Model.state_dict()
            for client in Clients_finetune:
                client.model.load_state_dict(global_dict)
            cLoss = [[] for _ in range(4)]
            for _, gparam in Global_Model.named_parameters():
                gparam.data.zero_()
            for client in Clients_finetune:
                modelfile = model_path + 'model_' + mod_str + '_' + exp_times + '.pth'
                print(modelfile)
                tmpLoss = client.train(modelfile, localepoch)
                for j in range(4):
                    cLoss[j].append(tmpLoss[j])
                for gParam, cParam in zip(Global_Model.parameters(), client.model.parameters()):
                    gParam.data += cParam.data / cnum
            cLoss = [np.mean(cLoss[i]) for i in range(4)]
            print(cLoss)
        torch.save(Global_Model.state_dict(), './model/finetune_' + classific_mode + '_' + exp_times + '.pth')
        tloss = []
        tacc = []
        for client in Clients_finetune:
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
    for i in range(1, experiment_times+1):
        fedTask_mix(fname_list, target, clientnum, mode=mode, exp_times=i)
