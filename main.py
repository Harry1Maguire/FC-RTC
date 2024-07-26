import pickle
import pcap_proc4
import pcap_proc6
from process import *

fname = 'iptas_drop1'  # 数据集文件夹名
ipv = 4                 # 数据集ip协议版本
target = 7              # 目标分类数目
clientnum = 5           # 客户端数目
localepoch = 1          # 客户端训练轮数
globalepoch = 100        # 全局训练轮数
No = 10
def load_dict(hash_list_name):
    with open(hash_list_name, 'r') as f:
        dict1_str = f.readline()
        dict2_str = f.readline()
        dict3_str = f.readline()
    dict1 = eval(dict1_str)
    dict2 = eval(dict2_str)
    dict3 = eval(dict3_str)
    return dict1, dict2, dict3

def save_dict(dict1, dict2, dict3, hash_save_file):
    with open(hash_save_file, 'w') as f:
        f.write(str(dict1) + '\n')
        f.write(str(dict2) + '\n')
        f.write(str(dict3) + '\n')
    print('字典变量已保存到文件'+hash_save_file+'中。')

def fedTask(filename, tarnum, cnum, no):
    no = str(no)
    train_list = {}
    val_list = {}
    test_list = {}
    datasets = []
    hash_file = 'hash-exp-iptas1-' + no + '.txt'
    if not os.path.exists(f'{hash_file}.txt'):
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
            hash_save_file = 'hash-exp-iptas1-' + no + '.txt' # 存储不同重复实验时的训练集验证集测试集哈希列表信息
            save_dict(train_list, val_list, test_list, hash_save_file)
    with open(f"{filename}.pkl", "rb") as f:
        [streams, files, fdict] = pickle.load(f)
    dict1, dict2, dict3 = load_dict(hash_file)
    for obj in files:
        list1 = dict1[obj]
        list2 = dict2[obj]
        list3 = dict3[obj]
        datasets.append(list(stream_to_data_hashlist(streams[obj], fdict[obj], list1, list2, list3)))
    dataset = datasets[0].copy()
    for i in range(1, len(datasets)):
        dataset[0] = dataset[0] + datasets[i][0]
        dataset[1] = dataset[1] + datasets[i][1]
        dataset[2] = dataset[2] + datasets[i][2]
    Global_Model = myDISTILLER(tarnum).to(device)
    Clients = []
    for i in range(cnum):
        Clients.append(Client(
            i,
            [dataset[i].part(1./clientnum) for i in range(3)],
            tarnum,
        ))
    for e in range(globalepoch):
        global_dict = Global_Model.state_dict()
        for client in Clients:
            client.model.load_state_dict(global_dict)
        cLoss = [[] for _ in range(4)]
        for _, gparam in Global_Model.named_parameters():
            gparam.data.zero_()
        for client in Clients:
            tmpLoss = client.train(localepoch)
            for j in range(4):
                cLoss[j].append(tmpLoss[j])
            for gParam, cParam in zip(Global_Model.parameters(), client.model.parameters()):
                gParam.data += cParam.data / cnum
        cLoss = [np.mean(cLoss[i]) for i in range(4)]
        print(cLoss)
    torch.save(Global_Model.state_dict(), './exp-iptas1/model_iptas_'+ no +'.pth')
    tloss = []
    tacc = []
    for client in Clients:
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
    if not os.path.exists(f'{fname}.pkl'):
        DatasetGenerate(fname, ipv)
    for i in range(1, No+1):
        fedTask(fname, target, clientnum, no=i)
