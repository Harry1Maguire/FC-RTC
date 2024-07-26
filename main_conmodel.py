import pickle
import pcap_proc4
import pcap_proc6
from process import *
from conmodel_coldstart import *
from conmodel_finetune import *
from CNNmodel import *
from CNN_RNN_2a_model import *
from CNN2model import *
from DNN_PAYmodel import *
from DNN_HDRmodel import *
import torch
from supconloss import SupConLoss
from save_print import save_print_output

classific_mode = 'iptas'  # 分类模式 vpn/type/application/iptas
ipv = 4  # 数据集ip协议版本
target = 6  # 目标分类数目 分类模式 vpn（2）/type（6）/application（15）后面会改
clientnum = 5  # 客户端数目
localepoch = 1  # 客户端训练轮数
globalepoch = 100  # 全局训练轮数
mode = 'coldstart'  # 训练模式 coldstrat（从头训练模型） / finetune（调用有监督学习模型进行重训练）
model_path = './exp2-finetunemodel/'  # 预训练模型路径



experiment_times = 10  # 重复实验次数
model_name = 'Distiller' # 选择基模型   Distiller CNN CNN_RNN_2a DNN_PAY DNN_HDR CNN2model



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
elif classific_mode == 'iptas':
    target = 7
    mod_str = 'iptas'
    fname_list = ['iptas_drop1', 'iptas_drop5', 'iptas_drop10', 'iptas_drop15', 'iptas_drop20']
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

# FC-RTC训练架构
def fedTask_mix(filename_list, tarnum, cnum, mode, exp_times):
    train_list = {}
    val_list = {}
    test_list = {}
    datasets = []
    exp_times = str(exp_times)
    hash_file = 'hash-num-' + exp_times
    if not os.path.exists(f'{hash_file}.txt'):
        for filename in filename_list:
            #if filename == filename_list[0]:  # 完整流量数据文件
            print(filename)
            with open(f"{filename}.pkl", "rb") as f:
                [streams, files, fdict] = pickle.load(f)
            for obj in files:
                print(obj)
                list1, list2, list3 = creat_hash_for_datasplit(streams[obj]) # 打乱训练数据，分为训练集验证集测试集
                datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
                train_list.setdefault(obj, list1)
                val_list.setdefault(obj, list2)
                test_list.setdefault(obj, list3)
            hash_save_file = 'hash-exp2-iptas-' + exp_times + '.txt' # 存储不同重复实验时的训练集验证集测试集哈希列表信息
            save_dict(train_list, val_list, test_list, hash_save_file)
            '''''''''
            else:
                print(filename)
                with open(f"{filename}.pkl", "rb") as f:
                    [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    print(obj)
                    list1 = train_list[obj]
                    list2 = val_list[obj]
                    list3 = test_list[obj]
                    datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
            '''
    else:
        for filename in filename_list:
            with open(f"{filename}.pkl", "rb") as f:
                [streams, files, fdict] = pickle.load(f)
                for obj in files:
                    dict1, dict2, dict3 = load_dict(hash_file + '.txt')
                    list1 = dict1[obj]
                    list2 = dict2[obj]
                    list3 = dict3[obj]
                    datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
    dataset = datasets[0].copy()
    for i in range(1, len(datasets)):
        dataset[0] = dataset[0] + datasets[i][0]
        dataset[1] = dataset[1] + datasets[i][1]
        dataset[2] = dataset[2] + datasets[i][2]

    if mode == 'coldstart':
        if model_name == 'Distiller':
            Global_Model = myDISTILLER_coldstart(tarnum).to(device)
            Clients_coldstart = []
            for i in range(cnum):
                Clients_coldstart.append(Client_coldstart(
                    i,
                    [dataset[i].part(1. / clientnum) for i in range(3)],
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
            torch.save(Global_Model.state_dict(), './model/normal_iptas1'+ classific_mode +'_' + exp_times +'.pth')
            tloss = []
            tacc = []
            tpre = []
            f1 = []
            for client in Clients_coldstart:
                tmp = client.test()
                tloss.append(tmp[0])
                tacc.append(tmp[1])
                tpre.append(tmp[2])
                f1.append(tmp[3])
            tloss = np.mean(tloss)
            tacc = np.mean(tacc)
            tpre = np.mean(tpre)
            f1 = np.mean(f1)
            print([tloss, tacc, tpre, f1])
        elif model_name == 'CNN':
            Global_Model = traffic_CNN(tarnum).to(device)
            Clients_coldstart = []
            for i in range(cnum):
                Clients_coldstart.append(Client_CNN(
                    i,
                    [dataset[i].part(1. / clientnum) for i in range(3)],
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
            torch.save(Global_Model.state_dict(), './model/normal_'+ classific_mode +'_' + exp_times +'.pth')
            tloss = []
            tacc = []
            tpre = []
            f1 = []
            for client in Clients_coldstart:
                tmp = client.test()
                tloss.append(tmp[0])
                tacc.append(tmp[1])
                tpre.append(tmp[2])
                f1.append(tmp[3])
            tloss = np.mean(tloss)
            tacc = np.mean(tacc)
            tpre = np.mean(tpre)
            f1 = np.mean(f1)
            print([tloss, tacc, tpre, f1])
        elif model_name == 'CNN_RNN_2a':
            Global_Model = traffic_CNN_RNN(tarnum).to(device)
            Clients_coldstart = []
            for i in range(cnum):
                Clients_coldstart.append(Client_CNN_RNN(
                    i,
                    [dataset[i].part(1. / clientnum) for i in range(3)],
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
            torch.save(Global_Model.state_dict(), './model/normal_' + classific_mode + '_' + exp_times + '.pth')
            tloss = []
            tacc = []
            tpre = []
            f1 = []
            for client in Clients_coldstart:
                tmp = client.test()
                tloss.append(tmp[0])
                tacc.append(tmp[1])
                tpre.append(tmp[2])
                f1.append(tmp[3])
            tloss = np.mean(tloss)
            tacc = np.mean(tacc)
            tpre = np.mean(tpre)
            f1 = np.mean(f1)
            print([tloss, tacc, tpre, f1])
        elif model_name == 'CNN2model':
            Global_Model = traffic_CNN2(tarnum).to(device)
            Clients_coldstart = []
            for i in range(cnum):
                Clients_coldstart.append(Client_CNN2(
                    i,
                    [dataset[i].part(1. / clientnum) for i in range(3)],
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
            torch.save(Global_Model.state_dict(), './model/normal_' + classific_mode + '_' + exp_times + '.pth')
            tloss = []
            tacc = []
            tpre = []
            f1 = []
            for client in Clients_coldstart:
                tmp = client.test()
                tloss.append(tmp[0])
                tacc.append(tmp[1])
                tpre.append(tmp[2])
                f1.append(tmp[3])
            tloss = np.mean(tloss)
            tacc = np.mean(tacc)
            tpre = np.mean(tpre)
            f1 = np.mean(f1)
            print([tloss, tacc, tpre, f1])
        elif model_name == 'DNN_PAY':
            Global_Model = traffic_DNN_PAY(tarnum).to(device)
            Clients_coldstart = []
            for i in range(cnum):
                Clients_coldstart.append(Client_DNN_PAY(
                    i,
                    [dataset[i].part(1. / clientnum) for i in range(3)],
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
            torch.save(Global_Model.state_dict(), './model/normal_' + classific_mode + '_' + exp_times + '.pth')
            tloss = []
            tacc = []
            tpre = []
            f1 = []
            for client in Clients_coldstart:
                tmp = client.test()
                tloss.append(tmp[0])
                tacc.append(tmp[1])
                tpre.append(tmp[2])
                f1.append(tmp[3])
            tloss = np.mean(tloss)
            tacc = np.mean(tacc)
            tpre = np.mean(tpre)
            f1 = np.mean(f1)
            print([tloss, tacc, tpre, f1])
        elif model_name == 'DNN_HDR':
            Global_Model = traffic_DNN_HDR(tarnum).to(device)
            Clients_coldstart = []
            for i in range(cnum):
                Clients_coldstart.append(Client_DNN_HDR(
                    i,
                    [dataset[i].part(1. / clientnum) for i in range(3)],
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
            torch.save(Global_Model.state_dict(), './model/normal_' + classific_mode + '_' + exp_times + '.pth')
            tloss = []
            tacc = []
            tpre = []
            f1 = []
            for client in Clients_coldstart:
                tmp = client.test()
                tloss.append(tmp[0])
                tacc.append(tmp[1])
                tpre.append(tmp[2])
                f1.append(tmp[3])
            tloss = np.mean(tloss)
            tacc = np.mean(tacc)
            tpre = np.mean(tpre)
            f1 = np.mean(f1)
            print([tloss, tacc, tpre, f1])
        else:
            print('There is no such model at the moment.')
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
                tmpLoss = client.train(modelfile, localepoch)
                for j in range(4):
                    cLoss[j].append(tmpLoss[j])
                for gParam, cParam in zip(Global_Model.parameters(), client.model.parameters()):
                    gParam.data += cParam.data / cnum
            cLoss = [np.mean(cLoss[i]) for i in range(4)]
            print(cLoss)
        torch.save(Global_Model.state_dict(), './model/finetune_'+ classific_mode +'_'+ exp_times +'.pth')
        tloss = []
        tacc = []
        tpre = []
        f1 = []
        for client in Clients_finetune:
            tmp = client.test()
            tloss.append(tmp[0])
            tacc.append(tmp[1])
            tpre.append(tmp[2])
            f1.append(tmp[3])
        tloss = np.mean(tloss)
        tacc = np.mean(tacc)
        tpre = np.mean(tpre)
        f1 = np.mean(f1)
        print([tloss, tacc, tpre, f1])

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
